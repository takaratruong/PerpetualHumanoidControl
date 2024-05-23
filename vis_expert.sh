#!/bin/bash
#
#SBATCH --job-name="phc_expert_collect"
#SBATCH --output=slurm_out/collect-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --nodelist=move1

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory="$SLURM_SUBMIT_DIR
# echo "conda environment="$CONDA_DEFAULT_ENV

ulimit -n 5000
max_idxs=3628 # Number of motions in the motion lib # 11626
collect_start_idx=0
collect_step_idx=100
# obs_type="phc"
obs_type="ref"

act_noise=0.06
num_envs=100 #15
seed=424

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_shape_pnn_handstand"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_shape_pnn_handstand/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1" 
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset1" 
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset1/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1" 
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1_retry" 
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1_retry/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1_retry"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_ground2getup_failed3_subset1_retry/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_throw/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp/Humanoid.pth"
# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_sprints"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_second_sprints/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/limp_failed1_subset1"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/limp_failed1_subset1/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1_retry"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_handstands_failed3_subset1_retry/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_third_kicks_punches"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_third_kicks_punches/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_thrid_sprint"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_thrid_sprint/Humanoid.pth"
 
# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_basketball"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_basketball/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed1_actual"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_fourth_sitdown_failed1_actual/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3_retry"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3_retry/Humanoid.pth"


# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_cartwheels_failed3_subset3/Humanoid_00048000.pth"
# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml \
#         --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path}\
#         --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
#         --act_noise ${act_noise} \
#         --checkpoint ${checkpoint} --num_envs ${num_envs} --test --mode collect --im_eval --obs_type ${obs_type} # --rand_start



# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im.yaml \
#         --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path}\
#         --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
#         --checkpoint ${checkpoint} --num_envs ${num_envs} --test --mode collect --im_eval 

        
# python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml \
#         --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path output/phc_shape_mcp_iccv\
#         --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
#         --num_envs ${num_envs} --test --mode collect


echo "Done"


# ## SPECIAL CASE #
# Limp failed first half
# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp_failed1_subset1_retry"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp_failed1_subset1_retry/Humanoid.pth"

# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml \
#         --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path}\
#         --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
#         --checkpoint ${checkpoint} --num_envs ${num_envs} --test --mode collect --obs_type ${obs_type} --im_eval 


#  Limp failed second half
network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp_failed1_subset1_second_half"
checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_pnn_limp_failed1_subset1_second_half/Humanoid.pth"

python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml \
        --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_clipped_motion_train.pkl --network_path ${network_path}\
        --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
        --act_noise ${act_noise} \
        --checkpoint ${checkpoint} --num_envs ${num_envs} --test --mode collect --obs_type ${obs_type} --im_eval 

