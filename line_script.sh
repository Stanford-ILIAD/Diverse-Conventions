#!/bin/bash

python serial_trainer.py --num_env_steps 125000 --episode_length 1250 --pop_size 2 --xp_weight 0.15 --mp_weight 0.0 --lr 2.5e-4 --critic_lr 2.5e-4 --use_linear_lr_decay --env_name Line --env_length 2 --run_dir 1mp0vn --seed 1 --do_validation

python serial_trainer.py --num_env_steps 125000 --episode_length 1250 --pop_size 2 --xp_weight 0.15 --mp_weight 1.0 --lr 2.5e-4 --critic_lr 2.5e-4 --use_linear_lr_decay --env_name Line --env_length 2 --run_dir 1mp1vn --seed 1 --do_validation

python serial_trainer.py --num_env_steps 125000 --episode_length 1250 --pop_size 2 --xp_weight 0.15 --mp_weight 0.25 --lr 2.5e-4 --critic_lr 2.5e-4 --use_linear_lr_decay --env_name Line --env_length 2 --run_dir 1mp25vn --seed 1 --do_validation

python serial_trainer.py --num_env_steps 125000 --episode_length 1250 --pop_size 2 --xp_weight 0.15 --mp_weight 0.5 --lr 2.5e-4 --critic_lr 2.5e-4 --use_linear_lr_decay --env_name Line --env_length 2 --run_dir 1mp5vn --seed 1 --do_validation

python stat_trainer.py --num_env_steps 125000 --episode_length 1250 --pop_size 2 --loss_type ADAP --loss_param 0.05 --lr 2.5e-4 --critic_lr 2.5e-4 --use_linear_lr_decay --env_name Line --env_length 2 --run_dir 1mp5vn --seed 1 --do_validation
