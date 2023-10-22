# Diverse-Conventions
Exploring techniques to generate diverse conventions in multi-agent settings. The current algorithm can be found in the XD directory.

Use this branch for validating the results of the toy environments (Blind Bandits and Balance Beam).

## Installation
```
mkdir Diverse-Conventions
cd Diverse-Conventions
conda create --name DiverseConventionsToy python=3.10
conda activate DiverseConventionsToy
pip install setuptools==65.5.0 "wheel<0.40.0"
pip install gym==0.21.0
pip install stable-baselines3==1.7.0
git init
git remote add origin https://github.com/Stanford-ILIAD/Diverse-Conventions.git
git pull origin master
git submodule update --init --recursive
pip install -e .
cd PantheonRL
pip install -e .
pip install -e overcookedgym/human_aware_rl/overcooked_ai
cd ..
```

## Reproducing Results

### Blind Bandits (Tree) Environment

For the baseline (ADAP) and CoMeDi results, run:
```
./tree_script.sh
```

You can construct the plots from the paper using:

```
cd plots
python plot_results.py --loss_type ADAP --env_name Tree --name ADAP
python plot_results.py --env_name Tree --name CoMeDi
cd ..
```

### Balance Beam (Line) Environment

For the baseline (ADAP) and CoMeDi results, run:
```
./line_script.sh
```

You can construct the table from the paper using:

```
# SP Column
python testing.py LOAD --ego-load Line/results/1mp0vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp0vn/1/convention1/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp25vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp25vn/1/convention1/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp5vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp5vn/1/convention1/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp1vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp1vn/1/convention1/models/actor.pt --env_name Line

# XP Column (HS column is "fails", PX is "aligned")
python testing.py LOAD --ego-load Line/results/1mp0vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp0vn/1/convention0/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp25vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp25vn/1/convention0/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp5vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp5vn/1/convention0/models/actor.pt --env_name Line
python testing.py LOAD --ego-load Line/results/1mp1vn/1/convention1/models/actor.pt LOAD --partner-load Line/results/1mp1vn/1/convention0/models/actor.pt --env_name Line

# LS Column
python testing.py LOAD --ego-load Line/results/1mp0vn/1/convention1/models/actor.pt LEFT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp25vn/1/convention1/models/actor.pt LEFT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp5vn/1/convention1/models/actor.pt LEFT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp1vn/1/convention1/models/actor.pt LEFT --env_name Line

# RS Column
python testing.py LOAD --ego-load Line/results/1mp0vn/1/convention1/models/actor.pt RIGHT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp25vn/1/convention1/models/actor.pt RIGHT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp5vn/1/convention1/models/actor.pt RIGHT --env_name Line
python testing.py LOAD --ego-load Line/results/1mp1vn/1/convention1/models/actor.pt RIGHT --env_name Line
```

## Tree Environment
To train two conventions:
```
python serial_trainer.py --num_env_steps 4000 --pop_size 2 --xp_weight 0.5 --mp_weight 0.0 --lr 1e-4 --env_name Tree --seed 1
```

There is also an interactive (user agent) program. You can choose between RAND (partner moves randomly), SAFE (partner strives for reward of 1), RISKY (partner strives for reward of 3), and LOAD (trained partner). 

To play against the safe agent, run this:

```
python tree_cli.py SAFE
```

Run this for loading the first convention (after training):

```
python tree_cli.py LOAD --partner-load ./Tree/results/standard/1/convention0/models/actor.pt
```

To load the second convention (after training):
```
python tree_cli.py LOAD --partner-load ./Tree/results/standard/1/convention1/models/actor.pt
```

In general, you can run
```
python tree_cli.py LOAD --partner-load ./Tree/results/standard/[SEED#]/convention[CONVENTION#]/models/actor.pt
```

## Line Environment
To train two conventions:

```
python serial_trainer.py --num_env_steps 125000 --pop_size 2 --xp_weight 0.15 --mp_weight 0.5 --lr 2.5e-4 --critic_lr 2.5e-4 --episode_length 1250 --use_linear_lr_decay --env_length 2 --env_name Line --seed 1
```

There is also an interactive (user agent) program. You can choose between RAND (partner moves randomly), LEFT (partner is biased towards the left to break symmetries), RIGHT (partner is biased towards the right to break symmetries), and LOAD (trained partner). 

To play against the left agent, run this:

```
python numline_cli.py LEFT
```

Run this for loading the first convention (after training):

```
python numline_cli.py LOAD --partner-load ./Line/results/standard/1/convention0/models/actor.pt
```

To load the second convention (after training):
```
python numline_cli.py LOAD --partner-load ./Line/results/standard/1/convention1/models/actor.pt
```

In general, you can run
```
python numline_cli.py LOAD --partner-load ./Line/results/standard/[SEED#]/convention[CONVENTION#]/models/actor.pt
```

## Overcooked Environment
To train two conventions:

```
python serial_trainer.py --num_env_steps 400000 --pop_size 2 --xp_weight 0.5 --mp_weight 0.0 --lr 1e-4 --env_name Overcooked --episode_length 4000 --seed 1
```

To try out the website, run this:
```
python overcooked_env/flask_app.py --modelpath_p0 ./Overcooked/results/standard/1/convention0/models/actor.pt --modelpath_p1 ./Overcooked/results/standard/1/convention0/models/actor.pt --layout_name simple
```

In general, you can switch out convention0 with whichever conventions you want to test out.
