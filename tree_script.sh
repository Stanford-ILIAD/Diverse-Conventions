#!/bin/bash

for i in {1..10}; do
    python serial_trainer.py --num_env_steps 10000 --pop_size 2 --xp_weight 0.5 --mp_weight 0.0 --lr 2e-5 --critic_lr 2e-5 --env_name Tree --seed $i --do_validation
    python stat_trainer.py --num_env_steps 10000 --pop_size 2 --loss_type ADAP --loss_param 0.2 --lr 2e-5 --critic_lr 2e-5 --env_name Tree --seed $i --do_validation
done

wait
