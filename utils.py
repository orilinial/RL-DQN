import torch
import numpy as np
from collections import namedtuple
import random
import torchvision.transforms as T
from PIL import Image


def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    averaged = np.convolve(data_set, weights, mode='valid')

    pre_conv = []
    for i in range(1, periods):
        pre_conv.append(np.mean(data_set[:i]))

    averaged = np.concatenate([pre_conv, averaged])
    return averaged


def one_hot(state):
    states_num = 500
    one_hot_vec = torch.zeros(states_num)
    one_hot_vec[state] = 1
    return one_hot_vec.unsqueeze(0)


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


def complex_encoder(state):
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

    passenger_on_taxi = torch.tensor([passenger_on_taxi]).float()
    encoded_state = torch.cat((taxi_col_one_hot, taxi_row_one_hot, target_loc_one_hot, passenger_on_taxi))
    return encoded_state.unsqueeze(0)


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


transforms = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])


def get_screen(env, device):
    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.uint8)
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    screen = 1 - transforms(screen).unsqueeze(0).to(device)
    return screen
