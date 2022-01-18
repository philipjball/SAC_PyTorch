import random
from datetime import datetime
from argparse import ArgumentParser
from collections import deque

import os

from numpy.lib import utils
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio
import h5py

import dmc
from sac import SAC_Agent
from utils import make_checkpoint, timestep_to_transition, observation_to_state, obs_spec_to_dim, OfflineDataContainer, get_pixel_timestep


def train_agent_model_free(agent, env, params, experiment_dir) -> OfflineDataContainer:
    
    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 10000
    gif_interval = 50000
    save_interval = 250000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    save_model = params['save_model']
    total_steps = params['total_steps']
    save_offline_replay_dataset = params['save_offline_replay_dataset']
    offline_replay_dataset_size = params['offline_replay_dataset_size']
    pixel_hw = params['pixel_hw']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=experiment_dir)

    samples_per_step = env._num_repeats
    avg_length = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []
    episode_step = 0
    episode_reward = 0
    i_episode = 1
    time_step = env.reset()
    prev_time_step = time_step
    offline_data_container = OfflineDataContainer()
    offline_data_rewards = []

    while samples_number < total_steps:
        if time_step.last():
            i_episode += 1
            if save_offline_replay_dataset and (len(offline_data_container) < offline_replay_dataset_size):
                offline_data_container.add_timestep(get_pixel_timestep(time_step, env, pixel_hw=pixel_hw))
                offline_data_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            episode_rewards.append(episode_reward)
            time_step = env.reset()
            prev_time_step = time_step
            episode_step = 0
            episode_reward = 0

        take_random_action = samples_number < n_random_actions
        if save_offline_replay_dataset and (len(offline_data_container) < offline_replay_dataset_size):
            offline_data_container.add_timestep(get_pixel_timestep(time_step, env, pixel_hw=pixel_hw))
        state = observation_to_state(time_step.observation)
        action = agent.get_action(state, random=take_random_action)
        time_step = env.step(action)
        # a step happened, let's increment
        cumulative_log_timestep += 1
        cumulative_timestep += 1
        episode_step += 1
        samples_number += samples_per_step
        # convert to s,a,r,s'
        transition = timestep_to_transition(prev_time_step, time_step)
        agent.replay_pool.push(transition)
        prev_time_step = time_step
        episode_reward += time_step.reward
        # update if it's time
        if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
            q1_loss, q2_loss, pi_loss, a_loss = agent.optimize(update_timestep, state_filter=state_filter)
            n_updates += 1
        # logging
        if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
            writer.add_scalar('Loss/Q-func_1', q1_loss, n_updates)
            writer.add_scalar('Loss/Q-func_2', q2_loss, n_updates)
            writer.add_scalar('Loss/policy', pi_loss, n_updates)
            writer.add_scalar('Loss/alpha', a_loss, n_updates)
            writer.add_scalar('Values/alpha', np.exp(agent.log_alpha.item()), n_updates)
            avg_length = np.mean(episode_steps)
            running_reward = np.mean(episode_rewards)
            make_gif = cumulative_timestep % gif_interval == 0
            eval_reward = evaluate_agent(env, agent, state_filter, n_starts=n_evals, make_gif=make_gif, experiment_dir=experiment_dir, step=cumulative_timestep)
            writer.add_scalar('Reward/Train', running_reward, cumulative_timestep)
            writer.add_scalar('Reward/Test', eval_reward, cumulative_timestep)
            print('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
            episode_steps = []
            episode_rewards = []
        if cumulative_timestep % save_interval == 0 and save_model:
            make_checkpoint(agent, cumulative_timestep, experiment_dir=experiment_dir)

    return offline_data_container, int(max(offline_data_rewards))


def evaluate_agent(env, agent, state_filter, experiment_dir, step, n_starts=1, make_gif=True):
    reward_sum = 0
    frames = []
    for n in range(n_starts):
        time_step = env.reset()
        if make_gif and n == 0:
            frame = env.physics.render(
                height=256,
                width=256,
                camera_id=0
            )
            frames.append(frame)
        while not time_step.last():
            state = observation_to_state(time_step.observation)
            action = agent.get_action(state, state_filter=state_filter, deterministic=True, random=False)
            time_step = env.step(action)
            reward_sum += time_step.reward
            if make_gif and n == 0:
                frame = env.physics.render(
                    height=256,
                    width=256,
                    camera_id=0
                )
                frames.append(frame)
        if make_gif and n == 0:
            gif_path = os.path.join(experiment_dir, 'step{}_reward{}.mp4'.format(int(step), int(reward_sum / n_starts)))
            imageio.mimsave(str(gif_path), frames, fps=24)
    return reward_sum / n_starts


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah_run')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--total_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--save_offline_replay_dataset', dest='save_offline_replay_dataset', action='store_true')
    parser.add_argument('--offline_replay_dataset_size', type=int, default=100_000)
    parser.add_argument('--pixel_hw', type=int, default=64, choices=[64,84])
    parser.set_defaults(save_model=False)
    parser.set_defaults(save_offline_replay_dataset=False)

    args = parser.parse_args()
    params = vars(args)

    seed = params['seed']
    env_name = params['env']
    env = dmc.make(env_name, 2, seed)
    state_dim = obs_spec_to_dim(env.observation_spec())
    action_dim = int(env.action_spec().shape[-1])
    now = datetime.utcnow()
    date_str = now.strftime('%Y.%m.%d')
    time_str = now.strftime('%H%M%S')
    experiment_dir = os.path.join('runs', params['env'], 'seed{}'.format(params['seed']), date_str, time_str)
    offline_dataset_dir = os.path.join('offline_data', '_'.join([params['env'], 'medium_replay']), 'stochastic', 'seed{}'.format(params['seed']), '{}px'.format(params['pixel_hw']))

    agent = SAC_Agent(seed, state_dim, action_dim, lr=params['learning_rate'])

    offline_data_container, max_reward = train_agent_model_free(agent=agent, env=env, params=params, experiment_dir=experiment_dir)

    if params['save_offline_replay_dataset']:
        os.makedirs(offline_dataset_dir, exist_ok=True)
        mixed_dataset = offline_data_container.return_dict()
        file_name = os.path.join(offline_dataset_dir, 'shard_1_reward_{}.hdf5'.format(max_reward))
        with h5py.File(file_name, 'w') as shard_file:
            for k in mixed_dataset:
                shard_file.create_dataset(k, data=mixed_dataset[k], compression='gzip')
        print("Saved medium_replay dataset at {}".format(offline_dataset_dir))


if __name__ == '__main__':
    main()
