# Diverse-Conventions
Exploring techniques to generate diverse conventions in multi-agent settings. The current algorithm can be found in the XD directory.

## Installation
```
mkdir Diverse-Conventions
cd Diverse-Conventions
conda create --name DiverseConventions python=3.10
conda activate DiverseConventions
git init
git remote add origin https://github.com/Stanford-ILIAD/Diverse-Conventions.git
git pull origin master
git submodule update --init --recursive
pip install -e .
cd PantheonRL
pip install -e .
cd ..
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
