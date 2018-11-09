import copy
import os
import pickle
import operator

import numpy as np
from robo_rl.common.utils import print_heading


class Buffer:

    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, item):

        if len(self) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = copy.deepcopy(item)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        if batch_size > len(self):
            raise ValueError('Sampling batch size greater than buffer size')

        indices = np.random.randint(0, len(self), batch_size)
        return operator.itemgetter(*indices)(self.buffer)

    # info can be env_name and other details
    def save_buffer(self, path=None, info='env'):
        if path is None:
            path = f'buffer/{info}/'
        print_heading("Saving replay buffer")
        os.makedirs(path, exist_ok=True)
        pickle.dump(self.buffer, open(path + "buffer.pkl", "wb"))

    def load_buffer(self, path):
        self.buffer = pickle.load(open(path, "rb"))
