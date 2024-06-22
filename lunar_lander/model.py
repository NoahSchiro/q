import random
import math

import torch
from torch import nn
from torch.nn import functional as F

# Model we weill be training simply maps the observation space to the action space
# Given any possible observation about the environment, the model should decide
# the best action to take
class DQN(nn.Module):

    def __init__(self, n_observations, action_space, eps_start, eps_end, eps_decay):
        super(DQN, self).__init__()

        self.steps_done = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.action_space = action_space

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_space.n)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def select_action(self, state, device):

        # Get a random number
        sample = random.random()

        # Compute random action probability for this step
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # If we meet the threshold, use the policy nets reasoning, else
        # just take a random action
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.forward(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.action_space.sample()]], dtype=torch.long
            ).to(device)

