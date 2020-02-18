import random
import uuid
from argparse import ArgumentParser
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from sac import SAC_Agent
from utils import MeanStdevFilter, Transition, make_gif


def train_agent_model_free(agent, env, update_timestep, seed, log_interval, gif_interval, ep_steps=1000, n_random_actions=10000, use_statefilter=False):
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_update_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    samples = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps

    writer = SummaryWriter()

    while samples_number < 3e7:
        time_step = 0
        env.reset()
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        while (not done):
            cumulative_log_timestep += 1
            cumulative_update_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            nextstate, reward, done, _ = env.step(action)
            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            agent.replay_pool.push(Transition(state, action, reward, nextstate, real_done))
            state = nextstate
            if state_filter:
                state_filter.update(state)
            running_reward += reward
            # update if it's time
            if cumulative_update_timestep % update_timestep == 0 and cumulative_update_timestep > agent.batchsize:
                q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep, state_filter=state_filter)
                n_updates += 1
                writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
                writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
                writer.add_scalar('Loss/policy', pi_loss, n_updates)
                writer.add_scalar('Loss/alpha', a_loss, n_updates)
                writer.add_scalar('Values/alpha', np.exp(agent.log_alpha.item()), n_updates)
        cumulative_timestep += time_step
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(cumulative_log_timestep/log_episode)
            running_reward = int((running_reward/log_episode))
            eval_reward = evaluate_agent(env, agent, state_filter)
            writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
            writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
            samples.append(samples_number)
            print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
            cumulative_log_timestep = 0
            log_episode = 0
            running_reward = 0
        if i_episode % gif_interval == 0:
            make_gif(agent, env, cumulative_timestep, state_filter)


def evaluate_agent(env, agent, state_filter, n_starts=10):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state, state_filter=state_filter, deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.set_defaults(obs_filter=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env = gym.make(params['env'])
    # assume symmetric and uniform action scaling
    action_scale = env.action_space.high[0] if env.action_space.high[0] != 1 else None

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC_Agent(seed, state_dim, action_dim, action_scale=action_scale)

    train_agent_model_free(agent=agent,
                            env=env, 
                            update_timestep=params['update_every_n_steps'],
                            seed=seed,
                            log_interval=10,
                            gif_interval=100,
                            n_random_actions=params['n_random_actions'],
                            use_statefilter=params['obs_filter'])


if __name__ == '__main__':
    main()
