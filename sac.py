from collections import deque

import torch
import torch.nn as nn

from .utils import Transition


class ReplayPool(object):

    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.transitions = deque(maxlen=capacity)
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self.memory.append(transition)
        
    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self.memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self.memory))
    
    def __len__(self) -> int:
        return len(self.memory)


class MLPNetwork(nn.Module):
    


def make_double_q_function():


