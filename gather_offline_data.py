import random
from argparse import ArgumentParser

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
import h5py

import dmc
from sac import SAC_Agent
from utils import observation_to_state, obs_spec_to_dim, get_pixel_timestep, OfflineDataContainer


def get_offline_dataset_shard(env, agent: SAC_Agent, episodes: int = 50, action_dist: str = 'stochastic', pixel_hw: int = 64, random_actions=False):
    reward_sum = 0
    total_steps = 0
    time_step_container = OfflineDataContainer()
    if action_dist == 'stochastic':
        deterministic = False
        gaussian = False
    elif action_dist == 'deterministic':
        deterministic = True
        gaussian = False
    elif action_dist == 'gaussian':
        deterministic = True
        gaussian = True
    for _ in range(episodes):
        time_step = env.reset()
        while not time_step.last():
            state = observation_to_state(time_step.observation)
            action = agent.get_action(state, deterministic=deterministic, random=random_actions)
            time_step_container.add_timestep(get_pixel_timestep(time_step, env, pixel_hw=pixel_hw))
            total_steps += 1
            time_step = env.step(action)
            reward_sum += time_step.reward
        time_step_container.add_timestep(get_pixel_timestep(time_step, env, pixel_hw=pixel_hw))
    offline_data_shard = time_step_container.return_dict()
    return offline_data_shard, int(reward_sum/episodes)


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='cheetah_run')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--agent_checkpoint', type=str, default=None)
    parser.add_argument('--dataset_size', type=int, default=100000)
    parser.add_argument('--action_dist', type=str, default='stochastic', choices=['stochastic', 'deterministic', 'gaussian'])
    parser.add_argument('--dataset_type', type=str, choices=['random', 'medium', 'expert'])
    parser.add_argument('--pixel_hw', type=int, choices=[64, 84], default=64)
    parser.add_argument('--distracting', dest='distracting', action='store_true')
    parser.add_argument('--distracting_difficulty', type=str, choices=['easy', 'medium', 'hard'], default=None)
    parser.add_argument('--multitask', dest='multitask', action='store_true')
    parser.add_argument('--multitask_task_name', type=str, choices=['len', 'torso_length'])
    parser.add_argument('--multitask_level', type=int, choices=[1,2,3,4,5,6,7,8])
    parser.add_argument('--num_shards', type=int, default=4)
    parser.set_defaults(deterministic=False)
    parser.set_defaults(distracting=False)
    parser.set_defaults(multitask=False)
    args = parser.parse_args()
    params = vars(args)

    assert not (params['multitask'] and params['distracting']), "Can't do both multitask and distracting unfortunately"
    seed = params['seed']
    env_name = params['env']
    env = dmc.make(env_name, 2, seed, params['distracting'], params['distracting_difficulty'], params['multitask_level'], params['multitask_task_name'])
    state_dim = obs_spec_to_dim(env.observation_spec())
    action_dim = int(env.action_spec().shape[-1])
    if params['distracting']:
        dataset_type = '_'.join([params['dataset_type'], 'distracting', params['distracting_difficulty']])
    elif params['multitask']:
        dataset_type = '_'.join([params['dataset_type'], 'multitask', params['multitask_task_name'], str(params['multitask_level'])])
    else:
        dataset_type = params['dataset_type']
    offline_dataset_dir = os.path.join('offline_data', '_'.join([params['env'], '{}'.format(dataset_type)]), params['action_dist'], 'seed{}'.format(params['seed']), '{}px'.format(params['pixel_hw']))
    os.makedirs(offline_dataset_dir, exist_ok=True)

    agent = SAC_Agent(seed, state_dim, action_dim)

    if params['dataset_type'] == 'random':
        print("Generating random data, ignoring agent checkpoint")
        random_actions = True
    else:
        agent.load_checkpoint(params['agent_checkpoint'])
        random_actions = False

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_shards = params['num_shards']

    for i in range(num_shards):
        offline_data, reward = get_offline_dataset_shard(
            agent=agent,
            env=env,
            episodes=250,
            action_dist=params['action_dist'],
            pixel_hw = params['pixel_hw'],
            random_actions=random_actions
        )
        file_name = os.path.join(offline_dataset_dir, 'shard_{}_reward_{}.hdf5'.format(i+1, int(reward)))
        shard_file = h5py.File(file_name, 'w')
        for k in offline_data:
            shard_file.create_dataset(k, data=offline_data[k], compression='gzip')
        print("Saved shard {} out of {}".format(i+1, num_shards))

if __name__ == '__main__':
    main()