
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

sys.path.insert(0,'/move/u/takaraet/my_diffusion_policy')
# sys.path.insert(0,'/move/u/mpiseno/src/my_diffusion_policy')

from diffusion_policy.workspace.base_workspace import BaseWorkspace


COLLECT_Z = False

# # Motions that PHC fails to collect after multiple attempts
# MOTIONS_TO_BE_FILTERED = [
#     'handstand',
# ]

# def is_forbidden(fname):
#     if any([bad_motion.lower() in fname.lower() for bad_motion in MOTIONS_TO_BE_FILTERED]):
#         return True
    
#     return False

def load_policy(payload):
    from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
    from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    # hydra_cfg = payload['cfg']

    # # Instantiate model
    # model_cfg = {**hydra_cfg.policy.model}
    # del model_cfg['_target_']
    # ema_model = TransformerForDiffusion(**model_cfg)

    # # Instantiate noise scheduler
    # noise_scheduler_cfg = {**hydra_cfg.policy.noise_scheduler}
    # del noise_scheduler_cfg['_target_']
    # scheduler = DDPMScheduler(**noise_scheduler_cfg)

    # # Instantiate policy
    # policy_cfg = {**hydra_cfg.policy}
    # del policy_cfg['_target_']
    # policy_cfg['model'] = ema_model
    # policy_cfg['noise_scheduler'] = scheduler
    # policy = DiffusionTransformerLowdimPolicy(**policy_cfg)

    # # Load state dict from payload. The normalizer state dict is also handled here
    # policy.load_state_dict(payload['state_dicts']['ema_model'])

    # policy.to('cuda')
    # policy.eval()
    # return policy

    hydra_cfg = payload['cfg']
    
    hydra_cfg['task']['dataset']['zarr_path'] ='/move/u/takaraet/my_diffusion_policy/phc_data/v0.0/phc_data_v0.0.zarr' # probably no need to set dataset since the checkpoint will take care of the normalizer anyway
    cls = hydra.utils.get_class(hydra_cfg._target_)
    workspace = cls(hydra_cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # If using DiffusionPolicy 
    # get policy from workspace
    # policy = workspace.model
    # if hydra_cfg.training.use_ema:
    #     policy = workspace.ema_model
    # policy.to('cuda')
    # policy.eval()

    # # If using BET 
    # policy = workspace.policy 
    # policy.to('cuda')
    # policy.eval()

    # Using IBC 
    policy = workspace.model 
    # import ipdb; ipdb.set_trace() # Takara
    policy.to('cuda')
    policy.eval()
    return policy 


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
        self.mode = config['mode'] # Set mode ('collect' or 'diff' from command line)
        if self.mode == 'diff' or self.mode == 'eval':
            assert config['ckpt_path'] is not None
            self.ckpt_path = config['ckpt_path']
            self.m2t_map_path = None
            # self.ckpt_version = re.search(r'v\d\.\d', self.ckpt_path).group()
            # self.ckpt_epoch = int(re.search(r'\d*\.ckpt', self.ckpt_path).group()[:-len('.ckpt')])
            # self.m2t_map_path = config['m2t_map_path'] # Path to the " motion fname to text" map
            # self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]
            # self.data_split = re.search(r'(train|val|test){1}\.npz', self.m2t_map_path).group()[:-len('.npz')]
            
            # self.ckpt_epoch = int(self.ckpt_path.split('epoch_')[-1][:-len('.ckpt')])
            #self.ckpt_version = re.search(r'v\d\.\d', self.ckpt_path).group()
            #self.ckpt_epoch = int(re.search(r'\d*\.ckpt', self.ckpt_path).group()[:-len('.ckpt')])
            #self.m2t_map_path = config['m2t_map_path'] # Path to the " motion fname to text" map
            #self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]
            #self.data_split = re.search(r'(train|val|test){1}\.npz', self.m2t_map_path).group()[:-len('.npz')]
            
        # self.policy_name = os.environ.get('POLICY_NAME', None)
        # if self.mode == 'collect':
        #     assert self.policy_name is not None

        self.collect_start_idx = config['collect_start_idx'] # Starting index for collecting data
        self.collect_step_idx = config['collect_step_idx'] # how much the collect index increases by each time
        self.act_noise = config['act_noise'] # Action noise level
        self.obs_type = config['obs_type']
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
        
        if flags.rand_start:
            self.curr_stpes += 1
            return torch.tensor([int(self.curr_stpes > self.max_steps)])


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

            if (
                self.mode == 'collect' or
                (self.mode == 'eval' and self.obs_type == 'phc')
            ):
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()
            else:
                curr_max = max_steps
            
            curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            # print(self.terminate_state.sum()) 
            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                print(f'Terminated: {self.terminate_state.sum()}')
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())
        
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
                
                if self.mode == 'eval' or self.mode =='collect':
                    print('FINAL SUCCESS RATE', self.success_rate)
                    print(f'Failed texts: {self.failed_texts}')
                    # if self.mode == 'diff':
                    #     exit() 
                    # import ipdb; ipdb.set_trace() # Takara


                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist)[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all)[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all)[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all
                    gt_pos_all = self.gt_pos_all

                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, concatenate=False, object_arr=True)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)
                    # import ipdb; ipdb.set_trace() # Takara
                    #metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))

                exit() 

                    # Michael: For saving motion tracking statistics to disk
                #     self.pdp_type = 'PDP_hard'
                #     metric_dir = f'mt_metrics/{self.pdp_type}/'
                #     pathlib.Path(metric_dir).mkdir(parents=True, exist_ok=True)
                #     metric_fname = f'ckpt={self.ckpt_epoch}.txt'
                #     metric_path = os.path.join(metric_dir, metric_fname)
                #     success = ~self.terminate_memory[0]
                #     failed_names = self.motion_lib.curr_motion_keys[~success]
                #     with open(metric_path, 'w') as f:
                #         f.write(f'Success Rate: {self.success_rate}\n')
                #         f.write(str(metrics_print) + '\n')
                #         f.write('Failed:\n')
                #         for name in failed_names:
                #             f.write(name + '\n')

                #     exit()
                # else:
                #     if self.terminate_state.sum() == humanoid_env.num_envs:
                #         exit(1)
                # exit() 
                # done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.
                # #humanoid_env.forward_motion_samples()
                # self.terminate_state = torch.zeros(
                #     self.env.task.num_envs, device=self.device
                # )

                # self.pbar.update(1)
                # self.pbar.refresh()
                # self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                # if COLLECT_Z: self.zs = []
                # self.curr_stpes = 0
            


            # TAKARA
            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)
        


        if self.mode == 'diff': 
            done = torch.tensor([int(self.curr_stpes > self.max_steps)])

        if self.mode == 'eval' and self.obs_type == 'phc':
            humanoid_env = self.env.task

            #self.curr_stpes += 1
            # import ipdb; ipdb.set_trace() # Takara
            max_steps = humanoid_env._motion_lib.get_motion_num_steps() - 1 #.max() 
            done = self.curr_stpes >= max_steps  

        # if self.mode=='collect': 
        #     humanoid_env = self.env.task
        #     max_steps = humanoid_env._motion_lib.get_motion_num_steps() - 1 #.max() 
        #     done = self.curr_stpes >= max_steps  

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

        # import ipdb; ipdb.set_trace() # Takara
        
        # obs_store = np.zeros((self.env.num_envs, max_steps, 312)) # 312 for local obs , 576 for phc obs, 648 312 + 
        # obs_store = np.zeros((self.env.num_envs, max_steps, 480)) # for diff obs + ref obs

        if self.obs_type == 't2m':
            obs_store = np.zeros((self.env.num_envs, max_steps, 360)) # 360 for obs , 576 for phc obs whcih includes ref (first 360 is normal obs) 
        elif self.obs_type == 'phc':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576)) #  for phc obs 
        elif self.obs_type =='ref':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576)) #  for phc obs 
        else:   
            raise Exception('Invalid obs_type')

        act_store = np.zeros((self.env.num_envs, max_steps, 69)) 
        done_envs = np.zeros(self.env.num_envs,dtype=bool)            
        
        if self.mode == 'diff' or self.mode == 'eval':
            # load checkpoint       
            payload = torch.load(open(self.ckpt_path, 'rb'), pickle_module=dill)
            hydra_cfg = payload['cfg']
            
            policy = load_policy(payload)

            # hydra_cfg['task']['dataset']['zarr_path'] = '/move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.2.1/v1.2.1_AMASS_obs-phc_train/data_v1.2.1_train.zarr'
                      
        
        # NOTE: Keep hardcoded_text None to sample random texts from the m2t_map.
        # Set hardcoded_text to a string value if you want to manually specficy a text.
        hardcoded_text = None    
        # random.seed(25)   
        # hardcoded_text = 'a person walks backwards' 
        hardcoded_text = 'a persons walks straight backwards'
        # hardcoded_text = 'a person walks forward'
        # hardcoded_text = 'a persons walks straight forwards'
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

            if flags.rand_start:
                self.env.task._shift_character() 
                print('SHIFITINGS')

            if self.mode == 'diff' or self.mode == 'eval':
                # import ipdb; ipdb.set_trace() # Takara

                # obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs*0))] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                if self.obs_type == 't2m':
                    obs_deque = collections.deque([self.env.task.diff_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type == 'phc':
                    obs_deque = collections.deque([self.env.task.phc_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type =='ref':
                    obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs))] * hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                else:
                    raise Exception('Invalid obs_type')
            
            # Sample a text goal - Michael  
            # import ipdb; ipdb.set_trace() # Takara
            
            if self.obs_type != 'phc':
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
                        print(f'Setting zero text embeds for eval because this is motion tracking')
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
                    elif self.obs_type == 'phc':
                        observation = self.env.task.phc_obs
                    elif self.obs_type =='ref':
                        observation = np.hstack((self.env.task.diff_obs, self.env.task.ref_obs))
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

                            #### only for experts 
                            # action_clean = action.clone()
                            # action += torch.randn_like(action) * self.act_noise  # if using the lower level RL controllers 

                        self.env.task.use_noisy_action = True                        
                        # if observation.shape[0] != self.env.num_envs:
                        #     import ipdb; ipdb.set_trace() # Takara
                        #     assert False, 'A Bug has been reached, Observation shape does not match the number of envs'
                        #     import ipdb; ipdb.set_trace() # Takara

                        # obs_store[~done_envs, n,:] = observation[~done_envs,:]

                    if observation.shape[0] != self.env.num_envs:
                        assert False, 'A Bug has been reached, Observation shape does not match the number of envs'


                    obs_store[~done_envs, n,:] = observation[~done_envs,:]
                    obs_store[~done_envs, n,:] = observation[~done_envs,:]

                    if self.mode == 'diff' or self.mode == 'eval':
                        obs_deque.append(observation)

                        if self.env.task.text_input:
                            text_embed = encode_text(self.env.task.text_input, clip_model)
                            text_embeds = text_embed.repeat(self.env.num_envs, 1)

                        # import ipdb; ipdb.set_trace() # Takara
                        # print(self.env.task.text_input)
                        clean_traj = torch.ones(self.env.num_envs)
                        # action_dict = policy.predict_action(        
                        #     {'obs': torch.tensor(np.stack(list(obs_deque), 1))},
                        #     torch.as_tensor(text_embeds, device=self.device), clean_traj
                        # )

                        # for ibc 
                        # import ipdb; ipdb.set_trace() # Takara
                        action_dict = policy.predict_action(        
                            {'obs': torch.tensor(np.stack(list(obs_deque), 1))},
                            # torch.as_tensor(text_embeds, device=self.device), clean_traj
                        )


                        action = action_dict['action'][:,0,:] # if horizon =1 then use action_pred
                        
                        # if self.mode == 'eval':
                        #     assert self.env.num_envs == observation.shape[0]
                        #     obs_store[~done_envs, n, :] = observation[~done_envs, :]
                        
                    # Step the environment 
                    obs_dict, r, done, info = self.env_step(self.env, action)

                    # import ipdb; ipdb.set_trace() # Takara
                    # Collect Action here. The env_step goes from the heirarchical action to the actual torque, which we capture.  
                    # if self.mode != 'expert':

                    # CHANGE BACK
                    #####################################################################################################################
                    
                    if self.mode in ['collect', 'pert']:      
                        if len(self.env.task.mean_action.shape) > 1:
                            assert self.env.task.mean_action.shape[0] == self.env.num_envs, 'a bug has been reached'
                            act_store[~done_envs, n,:] = self.env.task.mean_action[~done_envs,:]
                        else:
                            # import ipdb; ipdb.set_trace() # Takara
                            act_store[~done_envs, n,:] = self.env.task.mean_action.reshape(1,-1)[~done_envs,:]

                        # act_store[~done_envs, n,:] = action_clean[~done_envs,:] 
                        # pass
                    
                    # if self.mode in ['diff']:
                    #     # import ipdb; ipdb.set_trace() # Takara
                    #     act_store[~done_envs, n,:] = action[~done_envs,:]

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
                    # print(termination_state)
                    # print(done_envs)
                    # print()
                    # self.terminate_state_eval[~done_envs] = torch.logical_or(termination_state[~done_envs], self.terminate_state_eval[~done_envs])
                    self.terminate_state_eval[~done_envs] = torch.logical_or(termination_state[~done_envs], self.terminate_state_eval[~done_envs])

                    # failed_names = self.motion_lib.curr_motion_keys[self.terminate_state_eval.nonzero(as_tuple=False).squeeze()]        

                    # print(f'Failed motions: {failed_names}')
                    
                    # print(termination_state)
                    # print(self.terminate_state_eval.sum())
                    
                    # print(done_envs)
                    # print()
                    # if self.terminate_state_eval.sum() > 0:
                    #     import ipdb; ipdb.set_trace() # Takara
                    
                    # if done_envs.sum() > 0:
                    #     import ipdb; ipdb.set_trace() # Takara

                    # print(self.terminate_state)
                    # print()
                    # if np.sum(self.terminate_state) >0 :
                    #     import ipdb; ipdb.set_trace() # Takara


                    # if self.mode =='diff':
                    #     cutoff= 60 
                    #     if n >=cutoff:
                    #         data_dir = f'collected_data/diffusion_examples/'
                    #         pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                    #         data_fname = f'data_{hardcoded_text.replace(" ", "_")}.npz'
                    #         data_path = os.path.join(data_dir, data_fname)
                            
                    #         assert np.sum(act_store[:,0:cutoff-1,:],(0,1,2)) !=0, 'all 0 actions'
                    #         assert np.sum(obs_store[:,:,:],(0,1,2)) !=0, 'all 0 observations'
                            
                    #         # import ipdb; ipdb.set_trace() # Takara
                    #         np.savez(data_path, obs=obs_store[:,0:cutoff-1,:], act=act_store[:,0:cutoff-1,:], 
                    #                  ep_len = None, #np.ones(obs_store.shape[0])* cutoff, 
                    #                  ep_name = None,
                    #                 )       
                    #         exit(0)
                            
                    #         # import ipdb; ipdb.set_trace() 
                    #         # Data saved 
                    

                    if self.mode=='collect':
                        
                        if flags.rand_start: 
                            # if self.curr_stpes == 20: 
                            #     import ipdb; ipdb.set_trace() # Takara
                            # We will collect 20 frames, but go to 50, this is to ensure that we only collect succesfull recoveries.  
                            if self.curr_stpes>=150:
                                num_collect = 25 
                                
                                data_dir = f'collected_data/rand_init_-{self.obs_type}_sigma={self.act_noise}_num_collect={num_collect}'
                                assert np.sum(act_store[:,:,:],(0,1,2)) !=0, 'all 0 actions'
                                assert np.sum(obs_store[:,:,:],(0,1,2)) !=0, 'all 0 observations'

                                pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                                end_idx = self.collect_start_idx + self.collect_step_idx
                                data_fname = f'phc_randStart_sigma={self.act_noise}_{self.collect_start_idx}-{end_idx}.npz'
                                data_path = os.path.join(data_dir, data_fname)
                                
                                # process data 

                                # obs=obs_store[:,:num_collect,:]
                                # act=act_store[:,:num_collect,:],
                                # ep_len=np.ones(self.env.num_envs)* num_collect, #self.motion_lib.get_motion_num_steps().cpu().numpy(),
                                # ep_name = self.motion_lib.curr_motion_keys,
                                # terminate= np.where(self.terminate_state)[0]

                                # import ipdb; ipdb.set_trace() # Takara

                                # if terminate.sum().item() > 0:
                                obs = obs_store[~self.terminate_state_eval]
                                act = act_store[~self.terminate_state_eval]
                                ep_names = self.motion_lib.curr_motion_keys[~self.terminate_state_eval]
                                
                                # import ipdb; ipdb.set_trace() # Takara
                                
                                # careful consideration for ep_len 
                                np.savez(
                                    data_path,
                                    obs=obs[:,:num_collect,:], 
                                    act=act[:,:num_collect,:],
                                    ep_len=np.ones(obs.shape[0])* num_collect, #self.motion_lib.get_motion_num_steps().cpu().numpy(),
                                    ep_name = ep_names,
                                    # terminate= np.where(self.terminate_state)[0]
                                )
                                
                                print(f'Saved data to {data_path}')
                                # failed = ''
                                # exit_code = int(failed != '')
                                # exit(exit_code)
                                exit()

                        if done_envs.all():
                        # if self.curr_stpes>=1300 :
                            print(self.terminate_memory)     
                            import ipdb; ipdb.set_trace() # Takara
                            # failure = self.motion_lib.curr_motion_keys[np.where(self.terminate_memory)[0]]
                            # print(f'Failed motions: {failure}')

                            failed = ''
                            if self.terminate_state.any().item():
                                failed = '_FAILED'
                                failed_idx = self.terminate_state.nonzero(as_tuple=False).squeeze()
                                failed_names = self.motion_lib.curr_motion_keys[failed_idx]
                                print(f'Failed motions: {failed_names}')

                                if isinstance(failed_names, str):
                                    failed_names = [failed_names]
                                
                                # # Filter failed motions that are infeasible
                                # if all([is_forbidden(name) for name in failed_names]):
                                #     failed = ''

                            # data_dir = f'collected_data/obs-{self.obs_type}_sigma={self.act_noise}'
                            # data_dir = f'collected_data/limp_failed_first_half_20_obs-{self.obs_type}_sigma={self.act_noise}'
                            # assert np.sum(act_store[:,:,:],(0,1,2)) !=0, 'all 0 actions'
                            # assert np.sum(obs_store[:,:,:],(0,1,2)) !=0, 'all 0 observations'

                            # pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            # end_idx = self.collect_start_idx + self.collect_step_idx
                            # #data_fname = f'phc_data_sigma={self.act_noise}_{self.collect_start_idx}-{end_idx}{failed}.npz'
                            # data_fname = f'collect_{self.collect_start_idx}-{end_idx}{failed}.npz'
                            # data_path = os.path.join(data_dir, data_fname)


                            # obs = obs_store[~self.terminate_state_eval]
                            # act = act_store[~self.terminate_state_eval]
                            # ep_names = self.motion_lib.curr_motion_keys[~self.terminate_state_eval]
                            # ep_lens = self.motion_lib.get_motion_num_steps().cpu().numpy()[~self.terminate_state_eval]  

                            # import ipdb; ipdb.set_trace() 


                            # np.savez(
                            #         data_path,
                            #         obs=obs, 
                            #         act=act,
                            #         ep_len=ep_lens,
                            #         ep_name = ep_names,
                            #         # terminate= np.where(self.terminate_state)[0]
                            # )
                        
                            # if self.terminate_state.sum().item() > 0:
                            #     term =np.where(self.terminate_memory)[0]
                            # else:
                            #     term = []
                            # import ipdb; ipdb.set_trace()

                            # np.savez(
                            #     data_path,
                            #     obs=obs_store, act=act_store,
                            #     ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(),
                            #     ep_name = self.motion_lib.curr_motion_keys,
                            #     terminate= np.where(self.terminate_state)[0]
                            # )
                            
                            # print(f'Saved data to {data_path}')

                            # exit_code = int(failed != '')
                            # exit(exit_code)



                            # Data saved 
                    elif self.mode == 'eval':
                        if done_envs.all():
                            failed_idx = self.terminate_state_eval.nonzero(as_tuple=False).squeeze()
                            failed_names = self.motion_lib.curr_motion_keys[failed_idx]
                            
                            print(f'Failed motions: {failed_names}')
                            print(f'num failed motions: {len(failed_names)}')

                            print(f'GOT TO END OF EVAL')

                            # data_dir = f'eval_data/{self.ckpt_version}_{self.data_split}/ckpt={self.ckpt_epoch}'
                            # pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            # end_idx = self.collect_start_idx + self.collect_step_idx
                            # data_fname = f'data_{self.collect_start_idx}-{end_idx}.npz'
                            # data_path = os.path.join(data_dir, data_fname)
                            # np.savez(
                            #     data_path,
                            #     obs=obs_store, act=act_store,
                            #     #ep_len=steps.cpu().numpy(),
                            #     ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(), # Keep track of original ep_len
                            #     ep_name=self.motion_lib.curr_motion_keys
                            # )

                            # print(f'Saved data to {data_path}')
                            # exit()
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
