# [Hierarchical RL by decomposing Action Space on Actor-Critic Methods](https://raghavm1.github.io/hierarchical-ddpg-projectpage/)

## Abstract

In Hierarchical RL, typically, different sub-policies handle different tasks that might be hierarchical in nature. Instead, we want to decompose the available multi-dimensional continuous action space across sub-policies while trying to learn an overall global task. We aim to experiment with variants of different hierarchies in environments pertaining to locomotion where multiple joints (our action space) are controlled. We aim to perform this study on actor-critic based algorithms, and experiment with a hybrid approach involving hierarchical and modular action policies.

This codebase has multiple branches, each branch contains code for a hierarchical approach -

- master - contains code for the monolithic (vanilla approach)
- modular
- hierarchical
- hybrid
- cheenti 🐜 - code for using the ant environment in PyBullet, but in hierarchical action splitting. There are different branches named cheenti<x> where x is a different approach.
- results - all results with plots and videos

To run this code, same commands as in Tonic RL can be used, for the DDPG algorithm. Our work only focuses on the DDPG algorithm.

## Environments

We use 2 environments 

- BipedalWalker-v3
- AntBullet-v0

# Instructions

## Install from source

Download and install Tonic:
```bash
git clone https://github.com/fabiopardo/tonic.git
pip install -e tonic/
```

Install TensorFlow or PyTorch, for example using:
```bash
pip install tensorflow torch
```

## Launch experiments

Use TensorFlow or PyTorch to train an agent, for example using:
```bash
python3 -m tonic.train \
--header 'import tonic.torch' \
--agent 'tonic.torch.agents.PPO()' \
--environment 'tonic.environments.Gym("BipedalWalker-v3")' \
--name PPO-X \
--seed 0
```

Snippets of Python code are used to directly configure the experiment. This is
a very powerful feature allowing to configure agents and environments with
various arguments or even load custom modules without adding them to the
library. For example:
```bash
python3 -m tonic.train \
--header "import sys; sys.path.append('path/to/custom'); from custom import CustomAgent" \
--agent "CustomAgent()" \
--environment "tonic.environments.Bullet('AntBulletEnv-v0')" \
--seed 0
```

By default, environments use non-terminal timeouts, which is particularly
important for locomotion tasks. But a time feature can be added to the
observations to keep the MDP Markovian. See the
[Time Limits in RL](https://arxiv.org/pdf/1712.00378.pdf) paper for more
details. For example:
```bash
python3 -m tonic.train \                                                                                  ⏎
--header 'import tonic.tensorflow' \
--agent 'tonic.tensorflow.agents.PPO()' \
--environment 'tonic.environments.Gym("Reacher-v2", terminal_timeouts=True, time_feature=True)' \
--seed 0
```

Distributed training can be used to accelerate learning. In Tonic, groups of
sequential workers can be launched in parallel processes using for example:
```bash
python3 -m tonic.train \
--header "import tonic.tensorflow" \
--agent "tonic.tensorflow.agents.PPO()" \
--environment "tonic.environments.Gym('HalfCheetah-v3')" \
--parallel 10 --sequential 100 \
--seed 0
```

## Plot results

During training, the experiment configuration, logs and checkpoints are
saved in `environment/agent/seed/`.

Result can be plotted with:
```bash
python3 -m tonic.plot --path BipedalWalker-v3/ --baselines all
```
Regular expressions like `BipedalWalker-v3/PPO-X/0`,
`BipedalWalker-v3/{PPO*,DDPG*}` or `*Bullet*` can be used to point to different
sets of logs.
The `--baselines` argument can be used to load logs from the benchmark. For
example `--baselines all` uses all agents while `--baselines A2C PPO TRPO` will
use logs from A2C, PPO and TRPO.

Different headers can be used for the x and y axes, for example to compare the
gain in wall clock time of using distributed training, replace `--parallel 10`
with `--parallel 5` in the last training example and plot the result with:
```bash
python3 -m tonic.plot --path HalfCheetah-v3/ --x_axis train/seconds --x_label Seconds
```

## Play with trained models

After some training time, checkpoints are generated and can be used to play
with the trained agent:
```bash
python3 -m tonic.play --path BipedalWalker-v3/PPO-X/0
```

Environments are rendered using the appropriate framework. For example, when
playing with DeepMind Control Suite environments, policies are loaded in a
`dm_control.viewer` where `Space` is used to start the interaction, `Backspace`
is used to start a new episode, `[` and `]` are used to switch cameras and
double click on a body part followed by `Ctrl + mouse clicks` is used to add
perturbations.

## Play with models from tonic_data

The `tonic_data` repository can be downloaded with:
```bash
git clone https://github.com/fabiopardo/tonic_data.git
```

The best seed for each agent is stored in `environment/agent` and can be
reloaded using for example:
```bash
python3 -m tonic.play --path tonic_data/tensorflow/humanoid-stand/TD3
```

<div align="center">
  <img src="data/images/humanoid_perturbation.jpg" width=80%><br><br>
</div>

The full benchmark plots are available
[here](https://github.com/fabiopardo/tonic_data/blob/master/images/benchmark.pdf).

They can be generated with:
```bash
python3 -m tonic.plot \
--baselines all \
--backend agg --columns 7 --font_size 17 --legend_font_size 30 --legend_marker_size 20 \
--name benchmark
```

Or:
```bash
python3 -m tonic.plot \
--path tonic_data/tensorflow \
--backend agg --columns 7 --font_size 17 --legend_font_size 30 --legend_marker_size 20 \
--name benchmark
```

And a selection can be generated with:
```bash
python3 -m tonic.plot \
--path tonic_data/tensorflow/{AntBulletEnv-v0,BipedalWalker-v3,finger-turn_hard,fish-swim,HalfCheetah-v3,HopperBulletEnv-v0,Humanoid-v3,quadruped-walk,swimmer-swimmer15,Walker2d-v3} \
--backend agg --columns 5 --font_size 20 --legend_font_size 30 --legend_marker_size 20 \
--name selection
```

<div align="center">
  <img src="data/images/selection.jpg" width=100%><br><br>
</div>

# Credit

This work is made possible by the mighty Fabio Pardo and his amazing work Tonic RL!

## Other code bases

Tonic was inspired by a number of other deep RL code bases. In particular,
[OpenAI Baselines](https://github.com/openai/baselines),
 [Spinning Up in Deep RL](https://github.com/openai/spinningup)
and [Acme](https://github.com/deepmind/acme).

## Cite

Like our work? Cite, share and subscribe (star)

```
@article{shaan2025hierarchical,
    title={Hierarchical RL by decomposing Action Space on Actor-Critic Methods},
    author={Aryaman Shaan, Garvit Banga, Raghav Mantri},
    year={2025}
}
```
