
import hydra
import dill
import glob
import os
import sys
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

# sys.path.insert(0,'/move/u/takaraet/diffusion_policy')
# from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.path.insert(0,'/move/u/takaraet/my_diffusion_policy')
# sys.path.insert(0,'/move/u/mpiseno/src/my_diffusion_policy')

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
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        # Michael ==
        self.mode = config['mode'] # Set mode ('collect' or 'diff' from command line)
        if self.mode == 'diff':
            self.m2t_map_path = config['m2t_map_path'] # Path to the " motion fname to text" map
            self.m2t_map = np.load(self.m2t_map_path, allow_pickle=True)['motion_to_text_map'][()]

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
    
        if flags.im_eval:

            humanoid_env = self.env.task
            
            # termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            termination_state = info["terminate"]
            # self._motion_lib = humanoid_env._motion_lib

            max_steps = 150

            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                # if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                #     bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                #     if (~self.terminate_state[:bound]).sum() > 0:
                #         if self.mode == 'collect':
                #             humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                #         else:
                #             curr_max = max_steps # humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                #     else:
                #         curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                # else:
                if True: # Michael - keeping indentation of previous code
                    if self.mode == 'collect':
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()
                    else:
                        curr_max = max_steps #humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                if self.mode == 'collect':
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()
                else:
                    curr_max = max_steps #humanoid_env._motion_lib.get_motion_num_steps().max()

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
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

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs


                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt

                num_evals = 10
                if (humanoid_env.start_idx + humanoid_env.num_envs >= num_evals):
                    print('FINAL SUCCESS RATE', self.success_rate)
                    print(f'Failed texts: {self.failed_texts}')
                    exit() 
                    
                    # terminate_hist = np.concatenate(self.terminate_memory)
                    # succ_idxes = np.nonzero(~terminate_hist[: num_evals])[0].tolist()

                    # pred_pos_all_succ = [(self.pred_pos_all[:num_evals])[i] for i in succ_idxes]
                    # gt_pos_all_succ = [(self.gt_pos_all[:num_evals])[i] for i in succ_idxes]

                    # pred_pos_all = self.pred_pos_all[:num_evals]
                    # gt_pos_all = self.gt_pos_all[:num_evals]

                    # # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    # failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: num_evals]]
                    # success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: num_evals]]
                    # # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    # if flags.real_traj:
                    #     pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                    #     gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                    #     pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                    #     gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    # metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    # metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    # metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    # metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    # print("------------------------------------------")
                    # print("------------------------------------------")
                    # print(f"Success Rate: {self.success_rate:.10f}")
                    # print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    # print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    # # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])
                    # print(self.config['network_path'])
                    # if COLLECT_Z:
                    #     zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
                    #     zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
                    #     joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))
                    
  

                    # # joblib.dump(np.concatenate(self.zs_all[: humanoid_env._motion_lib._num_unique_motions]), osp.join(self.config['network_path'], "zs.pkl"))

                    # joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    # joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    # print("....")

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.
                humanoid_env.forward_motion_samples()
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
            obs_store = np.zeros((self.env.num_envs, max_steps, 360)) # 312 for local obs , 576 for phc obs, 648 312 + 
        elif self.obs_type == 'phc':
            obs_store = np.zeros((self.env.num_envs, max_steps, 576)) #  for phc obs 
        else:
            raise Exception('Invalid obs_type')
        
        act_store = np.zeros((self.env.num_envs, max_steps, 69)) 
        done_envs = np.zeros(self.env.num_envs,dtype=bool)
        
        if self.mode == 'diff':
            
            checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/01.28.30_noise_only_final/checkpoints/epoch=7500-train_action_mse_error=0.002.ckpt'
            
            checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/10.57.53_noise_debug_h1/checkpoints/epoch=7950-train_action_mse_error=0.002.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/01.54.37_clean_only/checkpoints/epoch=7700-train_action_mse_error=0.003.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/14.07.31_action_masked_nob2/checkpoints/epoch=7950-train_action_mse_error=0.002.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/17.42.47_noise-clean_debug_shuffle/checkpoints/epoch=7800-train_action_mse_error=0.001.ckpt'
            # checkpoint = "/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/01.28.30_noise_only_final/checkpoints/epoch=7500-train_action_mse_error=0.002.ckpt"

#             checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.09/14.07.31_action_masked_nob2/checkpoints/epoch=7950-train_action_mse_error=0.002.ckpt'

            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/00.39.18_walking-diffpol_v0.3/checkpoints/'
#             # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/00.39.39_walking-diffpol_v0.4/checkpoints/'
#             # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/00.40.14_walking-diffpol_v0.5/checkpoints/epoch=7800-train_action_mse_error=0.006.ckpt'
            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/00.40.14_walking-diffpol_v0.6/checkpoints/epoch=7850-train_action_mse_error=0.004.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/09.57.52_v6_fixed/checkpoints/latest.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/14.01.21_combined_batch_test/checkpoints/latest.ckpt'
            
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/10.48.36_noisy_mask_fixed/checkpoints/latest.ckpt' #.76
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/14.43.23_cond_layers2/checkpoints/epoch=7650-train_action_mse_error=0.001.ckpt' #.8 
            
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/16.18.42_cond_layers2_noText/checkpoints/epoch=7600-train_action_mse_error=0.001.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/18.24.51_phc_obs/checkpoints/latest.ckpt' 
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/19.47.44_phc_obs/checkpoints/latest.ckpt' # 2obs

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/20.41.28_phc_obs_hor1/checkpoints/latest.ckpt'


            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/21.20.41_phc_obs_hor1/checkpoints/latest.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/21.33.04_diff_ref/checkpoints/latest.ckpt'

            # noise level exp
            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/16.30.40_walking-diffpol_v0.6/checkpoints/epoch=7950-train_action_mse_error=0.002.ckpt' #.099
            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/16.30.40_walking-diffpol_v0.5/checkpoints/epoch=7750-train_action_mse_error=0.003.ckpt'
            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/16.29.50_walking-diffpol_v0.4/checkpoints/epoch=7950-train_action_mse_error=0.002.ckpt'
            # checkpoint = '/move/u/mpiseno/src/my_diffusion_policy/data/outputs/2024.01.10/16.26.32_walking-diffpol_v0.3/checkpoints/epoch=6850-train_action_mse_error=0.002.ckpt' #.06
            # checkpoint ='/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.10/23.24.29_v3/checkpoints/epoch=7950-train_action_mse_error=0.001.ckpt'
            
            #debug exps
            # checkpoint= '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/06.39.49_batch2048/checkpoints/latest.ckpt' #3300
            # checkpoint= '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/06.51.11_v1.3_2048/checkpoints/latest.ckpt' #3300

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/07.10.40_v1.3_2048_0-200_cpy1/checkpoints/latest.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/07.34.04_v.3_2048/checkpoints/latest.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/08.03.48_v1.3_2048_0-200_cpy4/checkpoints/epoch=0800-train_action_mse_error=0.004 (copy).ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/08.03.48_v1.3_2048_0-200_cpy4/checkpoints/latest.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/08.03.48_v1.3_2048_0-200_cpy4/checkpoints/epoch=1800-train_action_mse_error=0.003.ckpt'
            
            # PHC Task 
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/08.46.29_v2.0/checkpoints/latest.ckpt' #tracking
            
            # Ref and motion conditioned 
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/10.39.35_v3.0_diffref/checkpoints/latest.ckpt'
            # checkpoint ='/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/11.08.15_v3.0_diffref_masked/checkpoints/latest.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/11.30.24_v3.0_diffref_masked/checkpoints/latest.ckpt'
            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/16.40.28_v3.1_diffref_refGlobal_fixMasked/checkpoints/latest.ckpt'

            # checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/17.22.39_v3.1_diffref_refLocal_maskApplied/checkpoints/latest.ckpt'
            
            checkpoint = '/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.11/18.28.40_v3.2_test/checkpoints/latest.ckpt'

            # load checkpoint       
            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

            hydra_cfg = payload['cfg']
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
        # hardcoded_text = None        
        hardcoded_text = 'a persons walks straight backwards'
        # hardcoded_text = 'a person walks forward'
        # hardcoded_text = 'a persons walks straight forwards'
        # hardcoded_text = 'a person walks in a clockwise circle.'
        # hardcoded_text = 'a person walks in a counter clockwise circle.'
        
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

            if self.mode == 'diff':
                # import ipdb; ipdb.set_trace() # Takara

                # obs_deque = collections.deque([np.hstack((self.env.task.diff_obs, self.env.task.ref_obs*0))] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                if self.obs_type == 't2m':
                    obs_deque = collections.deque([self.env.task.diff_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                elif self.obs_type == 'phc':
                    obs_deque = collections.deque([self.env.task.phc_obs] *hydra_cfg.policy.n_obs_steps, maxlen=hydra_cfg.policy.n_obs_steps)
                else:
                    raise Exception('Invalid obs_type')

            # Sample a text goal - Michael
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
                    obs_dict = self.env_reset(done_indices)
                    if COLLECT_Z: z = self.get_z(obs_dict)  
                    
                    if self.obs_type == 't2m':
                        observation = self.env.task.diff_obs    
                    elif self.obs_type == 'phc':
                        observation = self.env.task.phc_obs
                    else:
                        raise Exception('Invalid obs_type')
                    
                    # print(self.env.task.diff_obs.shape)
                    # print(self.env.task.ref_obs.shape)
                    # observation = np.hstack((self.env.task.diff_obs, self.env.task.ref_obs))
                    
                    # print(observation.shape)
                    if self.mode == 'collect':
                        if has_masks:
                            masks = self.env.get_action_mask()
                            action = self.get_masked_action(obs_dict, masks, is_determenistic)
                        else:
                            action = self.get_action(obs_dict, is_determenistic)

                        self.env.task.use_noisy_action = True
                        # index_store = self.env.task.progress_buf
                        
                        if observation.shape[0] == self.env.num_envs:
                            # obs_store[~done_envs, index_store[~done_envs]-1,:] = observation[~done_envs,:]
                            obs_store[~done_envs, n,:] = observation[~done_envs,:]
                    
                    if self.mode == 'diff':
                        obs_deque.append(observation)

                        clean_traj = torch.ones(self.env.num_envs)
                        action_dict = policy.predict_action(
                            {'obs': torch.tensor(np.stack(list(obs_deque), 1))},
                            torch.as_tensor(text_embeds, device=self.device), clean_traj
                        )

                        action = action_dict['action'][:,0,:] # if horizon =1 then use action_pred
                    
                    # Step the environment 
                    obs_dict, r, done, info = self.env_step(self.env, action)
                    
                    # Collect Action here. The env_step goes from the heirarchical action to the actual torque, which we capture.  
                    if self.mode == 'collect':      
                        if self.env.task.mean_action.shape[0] == self.env.num_envs:
                            # index_store[~done_envs]-1
                            # act_store[~done_envs, index_store[~done_envs]-1,:] = self.env.task.mean_action[~done_envs,:]
                            act_store[~done_envs, n,:] = self.env.task.mean_action[~done_envs,:]

                    cr += r
                    steps += 1

                    # Record failed language prompts
                    # if self.mode == 'diff' and self.terminate_state.sum() > 0:
                    #     if sampled_texts is not None:
                    #         terminated_idxs = torch.argwhere(self.terminate_state).squeeze().tolist()
                    #         if not isinstance(terminated_idxs, np.ndarray):
                    #             terminated_idxs = [terminated_idxs]
                    #         failed = sampled_texts[terminated_idxs]

                    #         for fail in failed:
                    #             # import ipdb; ipdb.set_trace() # Takara
                    #             self.failed_texts.add(str(fail))
                            
                    
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

                    if self.mode=='collect':
                        if done_envs.all():
                            failed = ''
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
                            
                            data_dir = f'collected_data/obs-{self.obs_type}_sigma={self.act_noise}'
                            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
                            end_idx = self.collect_start_idx + self.collect_step_idx
                            data_fname = f'phc_data_sigma={self.act_noise}_{self.collect_start_idx}-{end_idx}{failed}.npz'
                            data_path = os.path.join(data_dir, data_fname)
                            np.savez(
                                data_path,
                                obs=obs_store, act=act_store,
                                ep_len=self.motion_lib.get_motion_num_steps().cpu().numpy(),
                                ep_name = self.motion_lib.curr_motion_keys
                            )

                            print(f'Saved data to {data_path}')

                            exit_code = int(failed != '')
                            exit(exit_code)
                            # Data saved 

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
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
