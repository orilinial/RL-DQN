import torch
import numpy as np
from collections import namedtuple
import random


def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    averaged = np.convolve(data_set, weights, mode='valid')

    pre_conv = []
    for i in range(1, periods):
        pre_conv.append(np.mean(data_set[:i]))

    averaged = np.concatenate([pre_conv, averaged])
    return averaged


def one_hot(state, states_num):
    one_hot_vec = torch.zeros(states_num)
    one_hot_vec[state] = 1
    return one_hot_vec


def decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


def complex_encoder(state, state_num):
    taxirow, taxicol, passidx, destidx = decode(state)
    # transfer taxi and target to one_hot:
    taxi_col_one_hot = torch.zeros(5)
    taxi_col_one_hot[taxicol] = 1
    taxi_row_one_hot = torch.zeros(5)
    taxi_row_one_hot[taxirow] = 1

    target_loc_one_hot = torch.zeros(4)
    if passidx == 4:  # Passenger is on taxi, hence target is his destination
        passenger_on_taxi = 1
        target_loc_one_hot[destidx] = 1
    else:  # Passenger not on taxi, hence target is the passenger
        passenger_on_taxi = 0
        target_loc_one_hot[passidx] = 1

    # passenger_loc_one_hot = torch.zeros(5)
    # passenger_loc_one_hot[passidx] = 1
    # destination_loc_one_hot = torch.zeros(4)
    # destination_loc_one_hot[destidx] = 1
    # encoded_state = torch.cat((taxi_col_one_hot, taxi_row_one_hot, passenger_loc_one_hot, destination_loc_one_hot))

    passenger_on_taxi = torch.tensor([passenger_on_taxi]).float()
    encoded_state = torch.cat((taxi_col_one_hot, taxi_row_one_hot, target_loc_one_hot, passenger_on_taxi))
    return encoded_state


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    import gym
    env = gym.make('Taxi-v2')
    state = env.reset()
    encoder = other(state, 500)
    print(encoder)
    env.render()
    env.close()
