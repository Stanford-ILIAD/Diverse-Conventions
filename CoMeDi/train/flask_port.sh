#!/bin/bash

# rm /tmp/simplecookedcache

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

for r in five_by_five random1 random3 scenario1_s simple
do
    python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir sp --ai_name SP --full_dir results/$r/sp/1 --use_render
    
    python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir xp --ai_name XP --full_dir results/$r/xp/1/oracle_8 --use_render

    python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir mp --ai_name MP --full_dir results/$r/mp/1/oracle_8 --use_render

    python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir adap --ai_name ADAP --full_dir results/$r/baselines/ADAP/adap_8/1/oracle_8 --use_render
    
done


# for p in 8
# do
#     python serial_trainer.py --num_env_steps 1000000 --episode_length 200 --env_length 200 --use_linear_lr_decay --entropy_coef 0.0 --env_name overcooked --seed 1 --restored 0 --n_rollout_threads 50 --ppo_epoch 10 --cuda --layer_N 2 --hidden_size 64 --lr 1e-2 --critic_lr 1e-2 --over_layout $1 --run_dir xp --pop_size $p --xp_weight 0.25 --mp_weight 0.0

# done
