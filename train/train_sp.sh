#!/bin/bash

# rm /tmp/simplecookedcache

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

for r in random1 simple
do
    python trainer.py --num_env_steps 1000000 --pop_size 1 --episode_length 200 --env_length 200 --env_name overcooked --seed 1 --restored 0 --n_rollout_threads 50 --ppo_epoch 5 --cuda --layer_N 2 --hidden_size 64 --lr 5e-3 --critic_lr 5e-3 --over_layout $r --run_dir sp

    python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir sp --ai_name SP
done
