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


max_idxs=3628 # Number of motions in the motion lib # 11626
collect_start_idx=0
collect_step_idx=100
obs_type="phc"
act_noise=0.0
num_envs=1 #15
seed=424

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_cartwheel_iccv"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_cartwheel_iccv/Humanoid.pth"

# network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_1_iccv"
# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_1_iccv/Humanoid.pth"

network_path="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_iccv"
checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_iccv/Humanoid_handstand.pth"

# checkpoint="/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_4_iccv/Humanoid.pth" #/move/u/takaraet/PerpetualHumanoidControl/output/phc_prim_iccv/Humanoid_00000100.pth
# /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_copycat_take5_train.pkl

python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im.yaml \
        --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path}\
        --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
        --checkpoint ${checkpoint} --num_envs ${num_envs} --test --mode collect --im_eval 

# python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml \
#         --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path output/phc_shape_mcp_iccv\
#         --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz \
#         --num_envs ${num_envs} --test --mode collect


echo "Done"

# run_collection () {
#     start_idx=$1
#     end_idx=$(( start_idx + collect_step_idx ))
#     num_envs=$(( max_idxs - start_idx < collect_step_idx ? max_idxs - start_idx : collect_step_idx ))

#     status=1
#     tries=0
#     while [ ${status} -ne 0 -a ${tries} -lt 3 ];
#     do
#         # python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl \
#         #  --network_path output/phc_shape_mcp_iccv --test --num_envs ${num_envs} --epoch -1 --m2t_map_path /move/u/takaraet/my_diffusion_policy/phc_data/v0.0/motion_to_text_map_v0.0.npz --mode=pert --seed ${seed} --collect_start_idx ${start_idx} --collect_step_idx ${collect_step_idx} --obs_type ${obs_type} --act_noise ${act_noise} #--headless
        
       
#         # Run the Python command
#         # python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_copycat_take5_train.pkl --network_path output/phc_shape_mcp_iccv --test --epoch -1 --im_eval --headless --mode collect --act_noise ${act_noise} --seed ${seed} --num_envs ${num_envs}  --collect_start_idx ${start_idx} --collect_step_idx ${collect_step_idx} --obs_type ${obs_type}
#         status=$?
#         # Increment the seed
#         ((seed++))
#         ((tries++))
#     done
# }


# ### FOR ALL MOTIONS ###

# while [ ${collect_start_idx} -lt ${max_idxs} ];
# do
#     echo "Collection start index: $collect_start_idx"

#     # Increment the collect_start_idx
#     run_collection ${collect_start_idx}
#     ((collect_start_idx+=collect_step_idx))

#     echo "" # Just for an empty line for better readability
# done

### FOR FAILED MOTIONS ###

# Input start_idx of failed motions here e.g.
#failed_idx=(1050 1700 1950 2050 2350)
# failed_idx=(50 100 150 1700 3350)

# Iterate over the list
# for i in "${failed_idx[@]}"
# do
#     echo "Collection start index: $i"
#     run_collection ${i}
#     echo ""
# done