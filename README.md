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
```

## Results

Gets the insane HalfCheetah result:

![example](./assets/HC-reward.PNG)

## TODO

* Tidy up gradients to make code run even quicker
* Play around with different approaches to make learning faster (i.e., varying how often we train, parallelisation)
* Deal with envs that have early termination (not due to time limits)
* Deal with envs that have an action range that is not in the interval (-1,1)
* Test on other environments
* Make Gifs