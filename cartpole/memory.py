from collections import namedtuple, deque

import random

# Represents a single transition from (state, action) -> new_state and the associated reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Storage object for our memory. Deque is the natural data structure as we want to
# keep samples from the newer examples and discard older examples
class ReplayMemory:

    # Intialize the deque with capacity 
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    # Save a transition to the memory
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # Randomly sample [batch_size] number of transitions
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Return the length of the deque
    def __len__(self):
        return len(self.memory)


