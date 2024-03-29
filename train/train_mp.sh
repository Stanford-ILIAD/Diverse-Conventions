#!/bin/bash

# rm /tmp/simplecookedcache

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

for p in 8
do
    python serial_trainer.py --num_env_steps 1000000 --episode_length 200 --env_length 200 --use_linear_lr_decay --entropy_coef 0.0 --env_name overcooked --seed 1 --restored 0 --n_rollout_threads 50 --ppo_epoch 10 --cuda --layer_N 2 --hidden_size 64 --lr 1e-2 --critic_lr 1e-2 --over_layout $1 --run_dir mp --pop_size $p --xp_weight 0.25 --mp_weight 1.0

    # python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir sp --ai_name SP
done
