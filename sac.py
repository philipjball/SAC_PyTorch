from collections import deque
import itertools
import random

import torch
import torch.nn as nn

from .utils import Transition, replay_pool

class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


def make_double_q_function(state_dim):
    dual_q = tuple(MLPNetwork(input_dim=state_dim, output_dim=1) for i in range(2))
    return dual_q


class SAC_Agent:

    def __init__(self, seed, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=1, tau=5e-3, batchsize=256, update_interval=1):
        self.gamma = gamma
        self.K_epochs = K_epochs
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.tau = tau
        
        torch.manual_seed(seed)

        self.q_funcs = make_double_q_function(state_dim)
        self.target_q_funcs = tuple(copy.deepcopy(q_func) for q_func in self.q_funcs)

        self.policy = MLPNetwork(state_dim, action_dim)

        self.q_optimizers = tuple(torch.optim.Adam(q_func.parameters(), lr=lr) for q_func in self.q_funcs)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool()

    def update_target(self):
        """moving average update of target networks"""
        for target_q_func, q_func in zip(self.target_q_funcs, self.q_funcs):
            for target_q_param, q_param in zip(target_q_func.parameters(), q_func.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * target_q_param.data)

    def update_q_functions(self, Transitions):


    def optimize(self):
