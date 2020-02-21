# Soft Actor Critic in PyTorch

A relatively minimal PyTorch SAC implementation from scratch. Uses a numerically stable Tanh Transformation to implement action sampling and log-prob calculation.

## Quick Start

Simply run:

`python train_agent.py`

for default args. Changeable args are:
```
--env: String of environment name (Default: HalfCheetah-v2)
--seed: Int of seed (Default: 100)
--use_obs_filter: Boolean that is true when used (seems to degrade performance)
--update_every_n_steps: Int of how many env steps we take before optimizing the agent (Default: 1)
--n_random_actions: Int of how many random steps we take to 'seed' the replay pool (Default: 10000)
--save_model: Boolean that is true when used (saves model when GIFs are made, loading and running is left as an exercise for the reader (or until I get around to it))
```

## Results

Gets the insane HalfCheetah result, tops out around 17,000:

![example](./assets/HalfCheetah.png)
