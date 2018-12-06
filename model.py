import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.distributions import Categorical


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(DQN, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.hidden_to_output(out)
        out = self.dropout(out)
        return out


class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(Policy, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.hidden_to_output(out)
        out = self.softmax(out)
        return out

#
# class ActorCritic(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
#         super(ActorCritic, self).__init__()
#
#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
#
#         self.actor = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_outputs),
#             nn.Softmax(dim=1),
#         )
#
#     def forward(self, x):
#         value = self.critic(x)
#         probs = self.actor(x)
#         dist = Categorical(probs)
#         return dist, value