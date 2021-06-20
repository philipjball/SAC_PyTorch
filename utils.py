import itertools
import math
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from numpy.random import default_rng
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'real_done'))


class MeanStdevFilter():
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
    
    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean


class ReplayPool:

    def __init__(self, action_dim, state_dim, capacity=1e6):
        self.capacity = int(capacity)
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._pointer = 0
        self._size = 0
        self._init_memory()
        self._rng = default_rng()

    def _init_memory(self):
        self._memory = {
            'state': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'action': np.zeros((self.capacity, self._action_dim), dtype='float32'),
            'reward': np.zeros((self.capacity), dtype='float32'),
            'nextstate': np.zeros((self.capacity, self._state_dim), dtype='float32'),
            'real_done': np.zeros((self.capacity), dtype='bool')
        }

    def push(self, transition: Transition):

        # Handle 1-D Data
        num_samples = transition.state.shape[0] if len(transition.state.shape) > 1 else 1
        idx = np.arange(self._pointer, self._pointer + num_samples) % self.capacity

        for key, value in transition._asdict().items():
            self._memory[key][idx] = value

        self._pointer = (self._pointer + num_samples) % self.capacity
        self._size = min(self._size + num_samples, self.capacity)

    def _return_from_idx(self, idx):
        sample = {k: tuple(v[idx]) for k,v in self._memory.items()}
        return Transition(**sample)

    def sample(self, batch_size: int, unique: bool = True):
        idx = np.random.randint(0, self._size, batch_size) if not unique else self._rng.choice(self._size, size=batch_size, replace=False)
        return self._return_from_idx(idx)

    def sample_all(self):
        return self._return_from_idx(np.arange(0, self._size))

    def __len__(self):
        return self._size

    def clear_pool(self):
        self._init_memory()

    def initialise(self, old_pool):
        # Not Tested
        old_memory = old_pool.sample_all()
        for key in self._memory:
            self._memory[key] = np.append(self._memory[key], old_memory[key], 0)


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


# Code courtesy of JPH: https://github.com/jparkerholder
def make_gif(policy, env, step_count, state_filter, maxsteps=1000):
    envname = env.spec.id
    gif_name = '_'.join([envname, str(step_count)])
    state = env.reset()
    done = False
    steps = []
    rewards = []
    t = 0
    while (not done) & (t< maxsteps):
        s = env.render('rgb_array')
        steps.append(s)
        action = policy.get_action(state, state_filter=state_filter, deterministic=True)
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        action = action.reshape(len(action), )
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    clip.write_gif('gifs/{}.gif'.format(gif_name), fps=30)


def make_checkpoint(agent, step_count, env_name):
    q_funcs, target_q_funcs, policy, log_alpha = agent.q_funcs, agent.target_q_funcs, agent.policy, agent.log_alpha
    
    save_path = "checkpoints/model-{}.pt".format(step_count)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'log_alpha_state_dict': log_alpha
    }, save_path)
