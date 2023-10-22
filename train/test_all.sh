#!/bin/bash

export MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache
export TF_CPP_MIN_LOG_LEVEL=2
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONWARNINGS="ignore"

for M in simple random1 random3 unident_s random0
do
    echo $M
    echo "MP"
    python testing.py --env_name overcooked --seed 1 --over_layout $M --run_dir $M --n_rollout_threads 1000 --layer_N 2 --hidden_size 64 --use_render results/$M/mp/1/oracle_8 results/$M/mp/1/oracle_8
    
    echo "XP"
    python testing.py --env_name overcooked --seed 1 --over_layout $M --run_dir $M --n_rollout_threads 1000 --layer_N 2 --hidden_size 64 --use_render results/$M/xp/1/oracle_8 results/$M/xp/1/oracle_8
    
    echo "ADAP"
    python testing.py --env_name overcooked --seed 1 --over_layout $M --run_dir $M --n_rollout_threads 1000 --layer_N 2 --hidden_size 64 --use_render results/$M/baselines/ADAP/adap_8/1/oracle_8 results/$M/baselines/ADAP/adap_8/1/oracle_8
    
    echo "SP"
    python testing.py --env_name overcooked --seed 1 --over_layout $M --run_dir $M --n_rollout_threads 1000 --layer_N 2 --hidden_size 64 --use_render results/$M/mp/1/convention0 results/$M/mp/1/convention0

    # echo "BC"
    # python bc_vs_bc.py --env_name overcooked --seed 1 --over_layout $M --run_dir $M --n_rollout_threads 30 --layer_N 2 --hidden_size 64 --use_render

    echo "********************************************************************"
    # python torch_to_tfjs.py --env_name overcooked --seed 1 --n_rollout_threads 1 --layer_N 2 --hidden_size 64 --over_layout $r --run_dir sp --ai_name SP
done
