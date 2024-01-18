#!/bin/bash

max_idxs=363 # Number of motions in the motion lib
collect_step_idx=100
m2t_map_path="/move/u/mpiseno/src/my_diffusion_policy/phc_data/processed/v1.4/v1.4_KIT_obs-t2m_test/m2t_map_v1.4_test.npz"


run_evaluation () {
    ckpt_path=$1
    start_idx=$2

    seed=0
    start_idx=${collect_start_idx}
    num_envs=$(( max_idxs - start_idx < collect_step_idx ? max_idxs - start_idx : collect_step_idx ))

    # Run the Python command
    python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --network_path output/phc_shape_mcp_iccv --test --epoch -1 --im_eval --headless --motion_file /move/u/takaraet/PerpetualHumanoidControl/phc/data/amass/pkls/amass_copycat_take5_train.pkl \
    --collect_start_idx ${start_idx} --collect_step_idx ${collect_step_idx} \
    --mode eval \
    --act_noise 0.0 \
    --seed ${seed} \
    --num_envs ${num_envs} \
    --ckpt_path ${ckpt_path} \
    --m2t_map_path ${m2t_map_path} \
    --obs_type t2m

    # Increment the seed
    ((seed++))
}

checkpoints=(
    "/move/u/takaraet/my_diffusion_policy/data/outputs/2024.01.14/22.31.53_t2m_task_v1.4/checkpoints/checkpoint_epoch_2000.ckpt"
)

# Iterate over the list
for ckpt in "${checkpoints[@]}"
do
    echo "Checkpoint: $ckpt"
    collect_start_idx=0

    while [ ${collect_start_idx} -lt ${max_idxs} ];
    do
        echo "Collection start index: $collect_start_idx"

        # Increment the collect_start_idx
        run_evaluation ${ckpt} ${collect_start_idx}
        ((collect_start_idx+=collect_step_idx))

        echo "" # Just for an empty line for better readability
    done
    echo ""
done
