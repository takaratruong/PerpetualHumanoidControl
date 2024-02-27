#!/bin/bash
#
#SBATCH --job-name="hopefully this one works...."
#SBATCH --output=train_out/phc_expert_train-%j.out
#SBATCH --cpus-per-task=8
##SBATCH --mem-per-cpu=6G
#SBATCH --mem=20G
#SBATCH --gres=gpu:a5000:1
#SBATCH --account=move
#SBATCH --partition=move
# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "conda environment = "$CONDA_DEFAULT_ENV

ulimit -n 5000

obs_type="phc"
seed=424

name="cartwheels_failed3_subset3_retry"
network_path="output/phc_prim_pnn_${name}"
num_envs=500 #  500 

# srun bash -c "python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path} --headless --num_envs ${num_envs} --mode collect"
# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl --network_path ${network_path} --headless --num_envs ${num_envs} --mode collect


srun bash -c "python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file  /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl  --network_path ${network_path} --headless --mode collect  --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz  --num_envs ${num_envs}"
# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file  /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl  --network_path ${network_path} --headless --mode collect  --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz  --num_envs ${num_envs}


# SPECIAL ONE FOR SECOND HALF SPLITS
# python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file  /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_clipped_motion_train.pkl  --network_path ${network_path} --headless --mode collect  --m2t_map_path /move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_train/m2t_map_v1.4_train.npz  --num_envs ${num_envs}

echo "Done"
