# Madrona RL Environments

Implementation of various RL Environments in Madrona

## Requirements

To use Madrona with GPU, you need a CUDA version of at least 11.7 and a cmake version of at least 3.18. For these environments, you also need to have conda environments (miniconda/anaconda).

To install miniconda (from miniconda3 instructions):
```
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm miniconda3/miniconda.sh
miniconda3/bin/conda init bash
# restart shell afterwards
```


## Installation

```
conda create -n CoMeDi python=3.10
conda activate CoMeDi
pip install torch numpy tensorboard

git clone https://github.com/bsarkar321/CoMeDi
cd CoMeDi
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..

pip install -e .
```

NOTE: For cmake, you make need to specify the cuda tookit directory as follows:

```
cmake -D CUDAToolkit_ROOT=/usr/local/cuda-12.0 ..
```

## Running scripts

Training MAPPO on overcooked:

``` shell
MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache python trainer.py --num_env_steps 10000000 --pop_size 1 --episode_length 200 --env_length 200 --env_name overcooked --seed 1 --over_layout simple --run_dir simple_sp --restored 0 --n_rollout_threads 500 --ppo_epoch 5 --cuda --layer_N 2 --hidden_size 64 --lr 1e-2 --critic_lr 1e-2

MADRONA_MWGPU_KERNEL_CACHE=/tmp/simplecookedcache python trainer.py --num_env_steps 10000000 --pop_size 1 --episode_length 200 --env_length 200 --env_name overcooked --seed 1 --over_layout random1 --run_dir coord_sp --restored 0 --n_rollout_threads 500 --ppo_epoch 5 --cuda --layer_N 2 --hidden_size 64 --lr 1e-2 --critic_lr 1e-2
```

