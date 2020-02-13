import uuid
from collections import deque

import numpy as np
import pandas as pd
import torch

from model import Transition

def train_agent_model_free(ppo, ensemble_env, memory, update_timestep, seed, log_interval, ep_steps, start_states, start_real_states):
    # logging variables
    running_reward = 0
    running_reward_real = 0
    avg_length = 0
    time_step = 0
    cumulative_update_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    samples = []
    rewards = []
    n_starts = len(start_states)

    env_name = ensemble_env.unwrapped.spec.id

    state_filter = ensemble_env.state_filter

    half = int(np.ceil(len(start_real_states[0]) / 2))

    env = ensemble_env.real_env

    memory.clear_memory()

    while samples_number < 3e7:
        for reset, real_reset in zip(start_states, start_real_states):
            time_step = 0
            done = False
            env.reset()
            state = reset
            env.unwrapped.set_state(real_reset[:half], real_reset[half:])
            i_episode += 1
            log_episode += 1
            state = env.reset()
            state_filter.update(state)
            state = state_filter(state)
            done = False

            while (not done):
                cumulative_log_timestep += 1
                cumulative_update_timestep += 1
                time_step += 1
                samples_number += 1
                action = ppo.select_action(state_filter(state), memory)
                nextstate, reward, done, _ = env.step(action)
                state = nextstate
                state_filter.update(state)

                memory.rewards.append(np.array([reward]))
                memory.is_terminals.append(np.array([done]))

                running_reward += reward

                # update if it's time
                if cumulative_update_timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    cumulative_update_timestep = 0
                    n_updates += 1

            # logging
            if i_episode % log_interval == 0:
                subset_resets_idx = np.random.randint(0, n_starts, 10)
                subset_resets = start_states[subset_resets_idx]
                subset_resets_real = start_real_states[subset_resets_idx]
                avg_length = int(cumulative_log_timestep/log_episode)
                running_reward = int((running_reward_real/log_episode))
                actual_reward = test_agent(ppo, ensemble_env, memory, ep_steps, subset_resets, subset_resets_real, use_model=False)
                samples.append(samples_number)
                rewards.append(actual_reward)
                print('Episode {} \t Samples {} \t Avg length: {} \t Avg reward: {} \t Actual reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, running_reward, actual_reward, n_updates))
                df = pd.DataFrame({'Samples': samples, 'Reward': rewards})
                df.to_csv("{}.csv".format(env_name + '-ModelFree-Seed-' + str(seed)))
                cumulative_log_timestep = 0
                log_episode = 0
                running_reward = 0