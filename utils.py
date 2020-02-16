from collections import namedtuple, deque
import itertools
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate'))


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

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self._memory.append(transition)
        
    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

    def filter_sample(self, batch_size: int, state_filter: MeanStdevFilter):
        transitions = self.sample(batch_size)
        states = torch.Tensor(state_filter(transitions.state))
        actions = torch.Tensor(np.stack(transitions.action))
        reward = torch.Tensor(np.stack(transitions.reward))
        nextstate = torch.Tensor(state_filter(transitions.nextstate))
        return states, actions, reward, nextstate

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()


class SACDataSet(Dataset):

    def __init__(self, transitions: Transition, state_filter: MeanStdevFilter):
        self.states = torch.Tensor(state_filter(transitions.state))
        self.actions = torch.Tensor(np.stack(transitions.action))
        self.reward = torch.Tensor(np.stack(transitions.reward))
        self.nextstate = torch.Tensor(state_filter(transitions.nextstate))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.reward[index], self.nextstate[index]


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)