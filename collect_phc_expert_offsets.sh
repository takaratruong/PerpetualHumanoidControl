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
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
echo "conda environment = "$CONDA_DEFAULT_ENV

#max_idxs=11626
max_idxs=3628 # Number of motions in the motion lib
collect_start_idx=0
collect_step_idx=1000
obs_type="phc"
act_noise=0.02

amass_train_file="/move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl"
amass_test_file="/move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_test.pkl"
# kit_file=" /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_copycat_take5_train.pkl"


ulimit -n 10000

run_collection () {
    start_idx=$1
    end_idx=$(( start_idx + collect_step_idx ))
    num_envs=$(( max_idxs - start_idx < collect_step_idx ? max_idxs - start_idx : collect_step_idx ))

    status=1
    seed=542
    tries=0
    # while [ ${status} -ne 0 -a ${tries} -lt 1 ];
    # do
        # Run the Python command
    # python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file ${kit_file} \
    # --network_path output/phc_shape_mcp_iccv --test --epoch -1 --im_eval --headless --mode collect --act_noise ${act_noise} --seed ${seed} --num_envs ${num_envs}  --collect_start_idx ${start_idx} --collect_step_idx ${collect_step_idx} \
    # --obs_type ${obs_type}
    
    # python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl \
    # --network_path output/phc_shape_mcp_iccv --test --act_noise ${act_noise} --seed ${seed} --num_envs ${num_envs}  --collect_start_idx ${start_idx} --collect_step_idx ${collect_step_idx} \
    # --obs_type phc --mode collect --rand_start --im_eval
    python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_diffPol_train.pkl \
    --network_path output/phc_shape_mcp_iccv --test --num_envs ${num_envs} --epoch -1  --seed ${seed} --act_noise ${act_noise} --collect_start_idx ${start_idx}  --collect_step_idx ${collect_step_idx} \
    --obs_type phc --mode collect --rand_start --im_eval --headless 

    # status=$?
    ((seed++))
    # ((seed++))
        # ((tries++))
    # done
}


### FOR ALL MOTIONS ###

while [ ${collect_start_idx} -lt ${max_idxs} ];
do
    echo "Collection start index: $collect_start_idx"

    # Increment the collect_start_idx
    run_collection ${collect_start_idx}
    ((collect_start_idx+=collect_step_idx))

    echo "" # Just for an empty line for better readability
done


### FOR FAILED MOTIONS ###

# Input start_idx of failed motions here e.g.
#failed_idx=(1050 1700 1950 2050 2350)
# failed_idx=(50 500 550 650 750)

# # Iterate over the list
# for i in "${failed_idx[@]}"
# do
#     echo "Collection start index: $i"
#     run_collection ${i}
#     echo ""
# done