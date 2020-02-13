from collections import namedtuple, deque
import itertools
import random
from torch.utils.data import DataLoader, Dataset

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate'))


class ReplayPool:

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=capacity)
        
    def push(self, transition: Transition):
        """ Saves a transition """
        self._memory.append(transition)
        
    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

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

    def __init__(self, transitions: Transition):
        self.states = Transition.state
        self.actions = Transition.action
        self.nextstate = Transition.nextstate
        self.reward = Transition.reward

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.nextstate[index], self.reward[index]