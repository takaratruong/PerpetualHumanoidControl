
import hydra
import dill
import glob
import os
import sys
import re
import pdb
import random
import os.path as osp
import pathlib
sys.path.append(os.getcwd())

import numpy as np
import torch
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
    
import learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from uhc.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs
import ipdb
import collections
import clip 

#TAKARA
# sys.path.insert(0,'/move/u/takaraet/motion_mimic')
# from algs.diff_policy import DiffusionPolicy

#sys.path.insert(0,'/move/u/takaraet/diffusion_policy')
# from diffusion_policy.workspace.base_workspace import BaseWorkspace

#sys.path.insert(0,'/move/u/mpiseno/src/my_diffusion_policy')
sys.path.insert(0,'/move/u/mpiseno/src/my_diffusion_policy')

from diffusion_policy.workspace.base_workspace import BaseWorkspace

COLLECT_Z = False


# Motions that PHC fails to collect after multiple attempts
MOTIONS_TO_BE_FILTERED = [
    'handstand',
]

def is_forbidden(fname):
    if any([bad_motion.lower() in fname.lower() for bad_motion in MOTIONS_TO_BE_FILTERED]):
        return True
    
    return False


class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config): 
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_state_eval = torch.zeros(self.env.task.num_envs, device=self.device, dtype=torch.bool)

        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        # Michael ==
        self.obs_type = config['obs_type']
        self.mode = config['mode'] # Set mode ('collect' or 'diff' from command line)
        if self.mode == 'diff' or self.mode == 'eval':
            assert config['ckpt_path'] is not None
            self.ckpt_path = config['ckpt_path']
            # self.ckpt_version = re.search(r'v\d\.\d', self.ckpt_path).group()
            # self.ckpt_epoch = int(re.search(r'\d*\.ckpt', self.ckpt_path).group()[:-len('.ckpt')])
            # self.m2t_map_path = config['m2t_map_path'] # Path to the " motion fname to text" map
            # self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]
            # self.data_split = re.search(r'(train|val|test){1}\.npz', self.m2t_map_path).group()[:-len('.npz')]
            
            self.ckpt_version = re.search(r'v\d\.\d', self.ckpt_path).group()
            self.ckpt_epoch = int(re.search(r'\d*\.ckpt', self.ckpt_path).group()[:-len('.ckpt')])
            self.m2t_map_path = config['m2t_map_path'] # Path to the " motion fname to text" map
            self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]
            self.data_split = re.search(r'(train|val|test){1}\.npz', self.m2t_map_path).group()[:-len('.npz')]



        self.collect_start_idx = config['collect_start_idx'] # Starting index for collecting data
        self.collect_step_idx = config['collect_step_idx'] # how much the collect index increases by each time
        self.act_noise = config['act_noise'] # Action noise level
        
        # ==
        
        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0
        self.motion_lib = humanoid_env._motion_lib #181 motion_lib_base <----- 

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _post_step(self, info, done):
        super()._post_step(info)
        # return done

        if self.mode == 'pert':
            return torch.zeros(self.env.num_envs).to(self.device) #(torch.arange(self.env.num_envs)).to(self.device)


        if flags.im_eval:

            humanoid_env = self.env.task
            
            # termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            termination_state = info["terminate"]
            # self._motion_lib = humanoid_env._motion_lib

            if self.mode == 'diff':
                max_steps = 150
            elif self.mode == 'eval':
                max_steps =  humanoid_env._motion_lib.get_motion_num_steps().max() #250 for eval t2m 

            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            # if (~self.terminate_state).sum() > 0:
            #     max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
            #     curr_ids = humanoid_env._motion_lib._curr_motion_ids
            #     # if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
            #     #     bound = (max_possible_id == curr_ids).nonzero()[0] + 1
            #     #     if (~self.terminate_state[:bound]).sum() > 0:
            #     #         if self.mode == 'collect':
            #     #             humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
            #     #         else:
            #     #             curr_max = max_steps # humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
            #     #     else:
            #     #         curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
            #     # else:
            #     if True:
            #         if self.mode == 'collect':
            #             curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()
            #         else:
            #             curr_max = max_steps #humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

            #     if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            # else:
            #     if self.mode == 'collect':
            #         curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()
            #     else:
            #         curr_max = max_steps #humanoid_env._motion_lib.get_motion_num_steps().max()

            if (
                self.mode == 'collect' or
                (self.mode == 'eval' and self.obs_type == 'phc')
            ):
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()
            else:
                curr_max = max_steps

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1
            
            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                #self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())
                self.success_rate = (1 - np.concatenate(self.terminate_memory).mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt

                #num_evals = 10
                #if (humanoid_env.start_idx + humanoid_env.num_envs >= num_evals):
                if True:
                    print('FINAL SUCCESS RATE', self.success_rate)
                    if self.mode == 'diff' or self.mode == 'eval':
                        if self.obs_type != 'phc':
                            exit()
                    
                    terminate_hist = np.concatenate(self.terminate_memory)
                    #succ_idxes = np.nonzero(~terminate_hist[:num_evals])[0].tolist()
                    succ_idxes = np.nonzero(~terminate_hist)[0].tolist()

                    # pred_pos_all_succ = [(self.pred_pos_all[:num_evals])[i] for i in succ_idxes]
                    # gt_pos_all_succ = [(self.gt_pos_all[:num_evals])[i] for i in succ_idxes]
                    pred_pos_all_succ = [(self.pred_pos_all)[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all)[i] for i in succ_idxes]

                    # pred_pos_all = self.pred_pos_all[:num_evals]
                    # gt_pos_all = self.gt_pos_all[:num_evals]
                    pred_pos_all = self.pred_pos_all
                    gt_pos_all = self.gt_pos_all

                    # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    # failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: num_evals]]
                    # success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: num_evals]]

                    # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    # if flags.real_traj:
                    #     pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                    #     gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                    #     pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                    #     gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, concatenate=False, object_arr=True)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    #metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))

                    # Michael: For saving motion tracking statistics to disk
                    metric_dir = f'mt_metrics/{self.pdp_type}/ckpt={self.ckpt_epoch}_with_names'
                    metric_fname = f'metrics_{self.collect_start_idx}-{self.collect_start_idx + self.collect_step_idx}.npz'
                    pathlib.Path(metric_dir).mkdir(parents=True, exist_ok=True)
                    metric_path = os.path.join(metric_dir, metric_fname)
                    success = ~self.terminate_memory[0]
                    failed_names = self.motion_lib.curr_motion_keys[~success]
                    np.savez(
                        metric_path,
                        success=success,
                        failed_names=failed_names,
                        **metrics
                    )

                    exit()

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.
                #humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0

            # TAKARA
            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)
        

        if self.mode == 'diff': 
            done = torch.tensor([int(self.curr_stpes > self.max_steps)])

        if self.mode == 'eval' and self.obs_type == 'phc':
            humanoid_env = self.env.task

            self.curr_stpes += 1
            # import ipdb; ipdb.set_trace() # Takara
            max_steps = humanoid_env._motion_lib.get_motion_num_steps() - 1 #.max() 
            done = self.curr_stpes >= max_steps  
                
        return done

    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def run(self): 
        print('-'*50)
        # print(self.env)

        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        motion_lengths = self.motion_lib.get_motion_num_steps() 
        max_steps = self.motion_lib.get_motion_num_steps().max() 
        ep_lens = self.motion_lib.get_motion_num_steps() 

        print(f'Max steps: {max_steps}')

        # import ipdb; ipdb.set_trace() # Takara
        
        # obs_store = np.zeros((self.env.num_envs, max_steps, 312)) # 312 for local obs , 576 for phc obs, 648 312 + 
        # obs_store = np.zeros((self.env.num_envs, max_steps, 480)) # for diff obs + ref obs

        if self.obs_type == 't2m':
            obs_store = np.zeros((self.env.num_envs, max_steps, 360)) # 312 for local obs , 576 for phc obs, 648 312 + 
        elif self.obs_type == 'ref':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576))   #diff obs + ref obs
        elif self.obs_type == 'phc':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576)) #  for phc obs 
        elif self.obs_type =='ref':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576)) #  for phc obs 
        else:   
            raise Exception('Invalid obs_type')
        
        act_store = np.zeros((self.env.num_envs, max_steps, 69)) 
        done_envs = np.zeros(self.env.num_envs,dtype=bool)            
        
        # NOTE: MT eval
        if self.mode == 'diff' or self.mode == 'eval':
            # load checkpoint
            payload = torch.load(open(self.ckpt_path, 'rb'), pickle_module=dill)

            hydra_cfg = payload['cfg']
            # hydra_cfg['task']['dataset']['zarr_path'] ='/move/u/takaraet/my_diffusion_policy/phc_data/v0.0/phc_data_v0.0.zarr' # probably no need to set dataset since the checkpoint will take care of the normalizer anyway


            # import ipdb; ipdb.set_trace() # Takara  
            cls = hydra.utils.get_class(hydra_cfg._target_)
            workspace = cls(hydra_cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            
            # get policy from workspace
            policy = workspace.model
            if hydra_cfg.training.use_ema:
                policy = workspace.ema_model
            policy.to('cuda')
            policy.eval()
        
        # NOTE: Keep hardcoded_text None to sample random texts from the m2t_map.
        # Set hardcoded_text to a string value if you want to manually specficy a text.
        hardcoded_text = None    
        # random.seed(25)   
        # hardcoded_text = 'a person walks backwards' 
        # hardcoded_text = 'a persons walks straight backwards'
        # hardcoded_text = 'a person walks forward'
        hardcoded_text = 'a persons walks straight forwards'
        # hardcoded_text = 'a person walks in a clockwise circle.'
        # hardcoded_text = 'a person walks 4 steps and stops'
        # hardcoded_text = 'a person walks in a counter clockwise circle.'
        # hardcoded_text = 'stand still'
        clip_model = load_and_freeze_clip(device='cuda')

        ###########################################################################
        j = 0   
        # MAIN LOOP TAKARA
        print(f'Num envs: {self.env.num_envs}')
        print(f'Is deterministic: {is_determenistic}')

        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()
            if self.mode =='pert':
                self.env.task._generate_fall_states()

            if self.mode == 'diff' or self.mode == 'eval':
                # import ipdb; ipdb.set_trace() # Takara

                obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs*0))] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)

                # NOTE: MT eval
                if self.obs_type == 't2m':
                    obs_deque = collections.deque([self.env.task.diff_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type == 'ref':
                    obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs))] * hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type == 'phc':
                    obs_deque = collections.deque([self.env.task.phc_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type =='ref':
                    obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs*0))] * hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                else:
                    raise Exception('Invalid obs_type')
            
            # Sample a text goal - Michael 
            if self.obs_type != 'phc':
                # t2m stuff
                if self.mode == 'diff':
                    sampled_texts = None
                    if hardcoded_text is None:
                        text_embeds, sampled_texts = sample_text_embeds(
                            self.env.num_envs,
                            self.m2t_map, clip_model
                        )
                        print(f'Sampled texts: {sampled_texts}')
                    else:
                        text_embed = encode_text(hardcoded_text, clip_model)
                        text_embeds = text_embed.repeat(self.env.num_envs, 1)
                        print(f'Hardcoded text: {hardcoded_text}')
                elif self.mode == 'eval':
                    hardcoded_text = None
                    text_embeds, sampled_texts = sample_text_embeds_for_eval(
                        self.env.num_envs,
                        self.m2t_map, 
                        self.motion_lib.curr_motion_keys,
                        clip_model
                    )
                    print(f'Sampled texts: {sampled_texts}')
                else:
                    text_embed = encode_text(hardcoded_text, clip_model)
                    text_embeds = text_embed.repeat(self.env.num_envs, 1)
                    print(f'Hardcoded text: {hardcoded_text}')
            else:
                if self.mode == 'eval':
                    if self.obs_type == 't2m':
                        hardcoded_text = None
                        text_embeds, sampled_texts = sample_text_embeds_for_eval(
                            self.env.num_envs,
                            self.m2t_map, 
                            self.motion_lib.curr_motion_keys,
                            clip_model
                        )
                    else:
                        text_embeds = np.zeros((self.env.num_envs, 512))
                        sampled_texts = None
                else:
                    text_embeds = np.zeros((self.env.num_envs, 512))
                    sampled_texts = None

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)
            
            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False
            
            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices=[]
            ep_end_collect = [] 
            
            self.failed_texts = set()
            with torch.no_grad():       
                
                for n in range(self.max_steps): # TAKARA EDIT
                    if self.mode in ['collect', 'pert']:
                        obs_dict = self.env_reset(done_indices)
                    
                    if COLLECT_Z: z = self.get_z(obs_dict)  
                    
                    if self.obs_type == 't2m':
                        #self.env.task._compute_task_obs() # Michael
                        observation = self.env.task.diff_obs   
                    elif self.obs_type == 'ref':
                        observation = np.hstack((self.env.task.diff_obs, self.env.task.ref_obs)) 
                    elif self.obs_type == 'phc':
                        observation = self.env.task.phc_obs
                    elif self.obs_type =='ref':
                        observation = np.hstack((self.env.task.diff_obs, self.env.task.ref_obs*0))
                    else:
                        raise Exception('Invalid obs_type')
                    
                    # printdone_indices
                    # print(self.env.task.ref_obs.shape)

                    # print(observation.shape)
                    if self.mode in ['collect', 'pert']:
                        if has_masks:
                            masks = self.env.get_action_mask()
                            action = self.get_masked_action(obs_dict, masks, is_determenistic)
                        else:
                            
                            # import ipdb; ipdb.set_trace() # Takara
                            action = self.get_action(obs_dict, is_determenistic)
                            action_clean = action.clone()
                            action += torch.randn_like(action) * self.act_noise  # if using the lower level RL controllers 

                        self.env.task.use_noisy_action = True

                        if observation.shape[0] == self.env.num_envs:
                            obs_store[~done_envs, n,:] = observation[~done_envs,:]

                    if self.mode == 'diff' or self.mode == 'eval':
                        # is_determenistic = True
                        # if has_masks:
                        #     masks = self.env.get_action_mask()
                        #     action = self.get_masked_action(obs_dict, masks, is_determenistic)
                        # else:
                        #     action = self.get_action(obs_dict, is_determenistic)

                        # NOTE: MT eval
                        obs_deque.append(observation)

                        # if self.env.task.text_input:
                        #     text_embed = encode_text(self.env.task.text_input, clip_model)
                        #     text_embeds = text_embed.repeat(self.env.num_envs, 1)

                        # NOTE: MT eval
                        clean_traj = torch.ones(self.env.num_envs)
                        action_dict = policy.predict_action(        
                            {'obs': torch.tensor(np.stack(list(obs_deque), 1))},
                            torch.as_tensor(text_embeds, device=self.device), clean_traj
                        )

                        action = action_dict['action'][:,0,:] # if horizon =1 then use action_pred
                        if self.mode == 'eval':
                            assert self.env.num_envs == observation.shape[0]
                            obs_store[~done_envs, n, :] = observation[~done_envs, :]
                    
                    # Step the environment 
                    obs_dict, r, done, info = self.env_step(self.env, action)

                    # Collect Action here. The env_step goes from the heirarchical action to the actual torque, which we capture.  
                    # if self.mode != 'expert':
                    
                    # CHANGE BACK
                    #####################################################################################################################
                    # if self.mode in ['collect', 'pert']:      
                    #     if self.env.task.mean_action.shape[0] == self.env.num_envs:
                    #         act_store[~done_envs, n,:] = self.env.task.mean_action[~done_envs,:]
                    # else: 
                    # if action_clean.shape[0] == self.env.num_envs:
                    #     act_store[~done_envs, n,:] = action_clean[~done_envs,:]
                    
                    # print(self.terminate_state)
                    #####################################################################################################################
                    cr += r
                    steps[~done_envs] += 1
                
                    
                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        #time.sleep(self.render_sleep*2) # Does commenting this out make rendering faster? - Michael 

                    all_done_indices = done.nonzero(as_tuple=False)
                    
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    done_envs[done_indices] = True

                    termination_state = info["terminate"]
                    self.terminate_state_eval[~done_envs] = torch.logical_or(termination_state[~done_envs], self.terminate_state_eval[~done_envs])
                    #print(self.terminate_state_eval.tolist())
        
                    if self.mode=='collect':

                        if done_envs.all():
                            print(self.terminate_memory)     

                            failure = self.motion_lib.curr_motion_keys[np.where(self.terminate_memory)[0]]
                            print(f'Failed motions: {failure}')
                            
                            failed = ''
                            failed_names = []
                            if self.terminate_state.any().item():
                                failed = '_FAILED'
                                failed_idx = self.terminate_state.nonzero(as_tuple=False).squeeze()
                                failed_names = self.motion_lib.curr_motion_keys[failed_idx]
                                print(f'Failed motions: {failed_names}')

                                if isinstance(failed_names, str):
                                    failed_names = [failed_names]
                                
                                # Filter failed motions that are infeasible
                                if all([is_forbidden(name) for name in failed_names]):
                                    failed = ''


                            # data_dir = f'collected_data/obs-{self.obs_type}_sigma={0.06}'
                            # ep_lens = self.motion_lib.get_motion_num_steps()
                            # for i, motion_id in enumerate(self.motion_lib._curr_motion_ids):
                            #     motion_name = self.motion_lib.curr_motion_keys[i]
                            #     if motion_name in failed_names:
                            #         continue
                                
                            #     # Write the new data to disk
                            #     motion_id = motion_id.item()
                            #     file_start_idx = (motion_id // 100) * 100
                            #     file_end_idx = file_start_idx + 100
                            #     data_fname = f'phc_data_sigma={0.06}_{file_start_idx}-{file_end_idx}.npz'
                            #     data_path = os.path.join(data_dir, data_fname)
                            #     if os.path.exists(data_path):
                            #         data = np.load(data_path, allow_pickle=True)
                            #         idx = motion_id % 100
                            #         assert motion_name == data['ep_name'][idx]

                            #         new_obs = obs_store[i, :ep_lens[i]]
                            #         new_act = act_store[i, :ep_lens[i]]
                            #         new_ep_len = ep_lens[i]
                            #         new_ep_name = motion_name

                            #         data['obs'][idx, :ep_lens[i]] = new_obs
                            #         data['act'][idx, :ep_lens[i]] = new_act
                            #         data['ep_len'][idx] = new_ep_len
                            #         data['ep_name'][idx] = new_ep_name

                            #         np.savez(
                            #             data_path,
                            #             **data
                            #         )

                            
                            data_dir = f'collected_data/obs-{self.obs_type}_sigma={self.act_noise}'
                            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            end_idx = self.collect_start_idx + self.collect_step_idx
                            data_fname = f'phc_data_sigma={self.act_noise}_{self.collect_start_idx}-{end_idx}{failed}.npz'
                            data_path = os.path.join(data_dir, data_fname)
                            np.savez(
                                data_path,
                                obs=obs_store, act=act_store,
                                ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy()-1,
                                ep_name = self.motion_lib.curr_motion_keys
                            )

                            success_names = [name for name in self.motion_lib.curr_motion_keys if name not in failed_names]
                            handle_failed_names(failed_names, success_names, data_dir)


                            # NOTE: TAKARA code from merge =====
                            # # data_dir = f'collected_data/obs-{self.obs_type}_sigma={self.act_noise}'
                            # data_dir = f'collected_data/cartwheel_20_obs-{self.obs_type}_sigma={self.act_noise}'
                            # assert np.sum(act_store[:,:,:],(0,1,2)) !=0, 'all 0 actions'

                            # pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            # end_idx = self.collect_start_idx + self.collect_step_idx
                            # data_fname = f'phc_data_sigma={self.act_noise}_{self.collect_start_idx}-{end_idx}{failed}.npz'
                            # data_path = os.path.join(data_dir, data_fname)

                            # # if self.terminate_state.sum().item() > 0:
                            # #     term =np.where(self.terminate_memory)[0]
                            # # else:
                            # #     term = []
                            # # import ipdb; ipdb.set_trace()

                            # np.savez(
                            #     data_path,
                            #     obs=obs_store, act=act_store,
                            #     ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(),
                            #     ep_name = self.motion_lib.curr_motion_keys,
                            #     terminate= np.where(self.terminate_state)[0]
                            # )
                            # =====

                            print(f'Saved data to {data_path}')

                            exit_code = int(failed != '')
                            exit(exit_code)
                            # Data saved 
                    elif self.mode == 'eval':
                        if done_envs.all():
                            failed_idx = self.terminate_state_eval.nonzero(as_tuple=False).squeeze()
                            failed_names = self.motion_lib.curr_motion_keys[failed_idx]
                            
                            print(f'Failed motions: {failed_names}')
                            print(f'num failed motions: {len(failed_names)}')

                            print(f'GOT TO END OF EVAL')

                            data_dir = f'eval_data/{self.ckpt_version}_{self.data_split}/ckpt={self.ckpt_epoch}'
                            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            end_idx = self.collect_start_idx + self.collect_step_idx
                            data_fname = f'data_{self.collect_start_idx}-{end_idx}.npz'
                            data_path = os.path.join(data_dir, data_fname)
                            np.savez(
                                data_path,
                                obs=obs_store, act=act_store,
                                #ep_len=steps.cpu().numpy(),
                                ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(), # Keep track of original ep_len
                                ep_name=self.motion_lib.curr_motion_keys
                            )

                            # if self.obs_type == 't2m':
                            #     data_dir = f'eval_data/{self.ckpt_version}_{self.data_split}/ckpt={self.ckpt_epoch}'
                            #     pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            #     end_idx = self.collect_start_idx + self.collect_step_idx
                            #     data_fname = f'data_{self.collect_start_idx}-{end_idx}.npz'
                            #     data_path = os.path.join(data_dir, data_fname)
                            #     np.savez(
                            #         data_path,
                            #         obs=obs_store, act=act_store,
                            #         #ep_len=steps.cpu().numpy(),
                            #         ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(), # Keep track of original ep_len
                            #         ep_name=self.motion_lib.curr_motion_keys
                            #     )

                            #print(f'Saved data to {data_path}')
                            exit()

                            # Data saved
                    elif self.mode=='pert':
                        cutoff= 150 #150
                        if n >=cutoff:
                            data_dir = f'pert_data/phc_task_amass_ref_fixed/'
                            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            end_idx = self.collect_start_idx + self.collect_step_idx
                            data_fname = f'data_recovery_{self.collect_start_idx}-{end_idx}.npz'
                            data_path = os.path.join(data_dir, data_fname)
                            
                            # assert that the entire arry is not zero
                            # import ipdb; ipdb.set_trace() # Takara
                            assert np.sum(act_store[:,0:cutoff-1,:],(0,1,2)) !=0, 'all 0 actions'
                            # import ipdb; ipdb.set_trace() # Takara
                            np.savez(data_path, obs=obs_store[:,0:cutoff-1,:], act=act_store[:,0:cutoff-1,:], 
                                    #  ep_len = self.motion_lib.get_motion_num_steps().cpu().numpy(), 
                                    #  ep_name = self.motion_lib.curr_motion_keys,
                                     terminated = torch.argwhere(self.env.task.im_reward_track <.75).squeeze().numpy()) 

                            print('num of failed trajectories:', torch.argwhere(self.env.task.im_reward_track <.8).squeeze().numpy().shape[0])
                            exit(0)
                            
                            # import ipdb; ipdb.set_trace() 
                            # Data saved 

                    if done_count > 0:
                        
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        import ipdb; ipdb.set_trace() # Takara

        return True


def handle_failed_names(failed_names, success_names, data_dir):
    all_recorded_failed = []
    if os.path.exists(os.path.join(data_dir, 'failed.txt')):
        with open(os.path.join(data_dir, 'failed.txt'), 'r') as f:
            all_recorded_failed = f.read().split('\n')
            all_recorded_failed = [name for name in all_recorded_failed if name != '']

    # Handle names that previously failed but are now successful
    for name in success_names:
        if name in all_recorded_failed:
            all_recorded_failed.remove(name)
    
    # Handle new failed names
    for name in failed_names:
        if name not in all_recorded_failed:
            all_recorded_failed.append(name)
    
    with open(os.path.join(data_dir, 'failed.txt'), 'w') as f:
        f.write('\n'.join(all_recorded_failed) + '\n')


def clean_raw_text(raw_text):
    all_annotations = raw_text.split('\n')
    if '' in all_annotations:
        all_annotations.remove('')

    annot = random.choice(all_annotations)
    english = annot.split('#')[0]
    return english


def sample_text_embeds(num_samples, m2t_map, clip_model):
    text_embeds = []
    texts = []
    for _ in range(num_samples):
        motion_file = random.choice(list(m2t_map.keys()))
        raw_text = m2t_map[motion_file]
        text = clean_raw_text(raw_text)
        text_embed = encode_text(text, clip_model)
        text_embeds.append(text_embed)
        texts.append(text)

    text_embeds = np.vstack(text_embeds) 
    texts = np.array(texts)
    return text_embeds, texts


def sample_text_embeds_for_eval(num_samples, m2t_map, motion_keys, clip_model):
    '''
    Different from sample_text_embeds because we sample texts from motion keys instead of 
    just from the m2t_amp
    '''
    text_embeds = []
    texts = []
    assert num_samples == len(motion_keys) # Make sure number of environemtns is the same as number of motions
    for idx in range(len(motion_keys)):
        motion_file = motion_keys[idx]
        if '0-' in motion_file:
            motion_file = motion_file.replace('0-', '')

        raw_text = m2t_map[motion_file]
        text = clean_raw_text(raw_text)
        text_embed = encode_text(text, clip_model)
        text_embeds.append(text_embed)
        texts.append(text)

    text_embeds = np.vstack(text_embeds) 
    texts = np.array(texts)
    return text_embeds, texts


def load_and_freeze_clip(clip_version='ViT-B/32', device='cuda'):
    clip_model, clip_preprocess = clip.load(clip_version, device=device)
    if str(device) != 'cpu':
        clip.model.convert_weights(clip_model)

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model

def encode_text(text, model):
    tokens = clip.tokenize(text).to('cuda')
    encoding = model.encode_text(tokens).float()
    return encoding
