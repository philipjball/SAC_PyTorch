import uuid
from collections import deque

import gym
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import Transition, MeanStdevFilter
from sac import SAC_Agent

def train_agent_model_free(agent, env, update_timestep, seed, log_interval, ep_steps, start_states, start_real_states):
    # logging variables
    running_reward = 0
    running_reward_real = 0
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
    rewards = []
    n_starts = len(start_states)

    state_filter = MeanStdevFilter(env.env.observation_space.shape[0])

    state_filter.mean = np.zeros(state_filter.shape)
    state_filter.stdev = np.ones(state_filter.shape)

    half = int(np.ceil(len(start_real_states[0]) / 2))

    writer = SummaryWriter()

    while samples_number < 3e7:
        # for reset, real_reset in zip(start_states, start_real_states):
        time_step = 0
        env.reset()
        # state = reset
        # env.unwrapped.set_state(real_reset[:half], real_reset[half:])
        i_episode += 1
        log_episode += 1
        state = env.reset()
        # state_filter.update(state)
        done = False

        while (not done):
            cumulative_log_timestep += 1
            cumulative_update_timestep += 1
            time_step += 1
            samples_number += 1
            action = agent.get_action(state_filter(state))
            nextstate, reward, done, _ = env.step(action)
            agent.replay_pool.push(Transition(state, action, reward, nextstate))
            state = nextstate
            # state_filter.update(state)
            running_reward += reward
            # update if it's time
            if cumulative_update_timestep % update_timestep == 0 and cumulative_update_timestep > agent.batchsize:
                q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(state_filter, update_timestep)
                cumulative_update_timestep = 0
                n_updates += 1
                writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
                writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
                writer.add_scalar('Loss/policy', pi_loss, n_updates)
                writer.add_scalar('Loss/alpha', a_loss, n_updates)
                writer.add_scalar('Values/alpha', np.exp(agent.log_alpha.item()), n_updates)
        cumulative_timestep += time_step
        # logging
        if i_episode % log_interval == 0:
            subset_resets_idx = np.random.randint(0, n_starts, 10)
            subset_resets = start_states[subset_resets_idx]
            subset_resets_real = start_real_states[subset_resets_idx]
            avg_length = int(cumulative_log_timestep/log_episode)
            running_reward = int((running_reward/log_episode))
            eval_reward = evaluate_agent(env, agent, state_filter)
            writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
            writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
            # actual_reward = test_agent(agent, ensemble_env, memory, ep_steps, subset_resets, subset_resets_real, use_model=False)
            samples.append(samples_number)
            # rewards.append(actual_reward)
            print('Episode {} \t Samples {} \t Avg length: {} \t Avg reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, running_reward, n_updates))
            # print('Episode {} \t Samples {} \t Avg length: {} \t Avg reward: {} \t Actual reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, running_reward, actual_reward, n_updates))
            # df = pd.DataFrame({'Samples': samples, 'Reward': rewards})
            # df.to_csv("{}.csv".format(env_name + '-ModelFree-Seed-' + str(seed)))
            cumulative_log_timestep = 0
            log_episode = 0
            running_reward = 0


def evaluate_agent(env, agent, state_filter):
    reward_sum = 0
    for i in range(10):
        done = False
        state = env.reset()
        while (not done):
            action = agent.get_action(state_filter(state), deterministic=True)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / 10           


def main():

    seed = 100

    env = gym.make('HalfCheetah-v2')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SAC_Agent(seed, state_dim, action_dim)

    env_resets = []
    env_resets_real = []

    for _ in range(1):
        env_resets.append(env.reset())
        env_resets_real.append(env.unwrapped.state_vector())
    
    env_resets = np.array(env_resets)
    env_resets_real = np.array(env_resets_real)

    train_agent_model_free(agent, env, 1, seed, 10, 1000, env_resets, env_resets_real)


if __name__ == '__main__':
    main()
