from utils import *
from model import ActorCritic
import numpy as np
import torch.optim as optim
import torch
import gym
from itertools import count
from eval_model import eval_model
import matplotlib
from torch.distributions import Categorical
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import argparse


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train_taxi_pg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    states_num = env.observation_space.n
    actions_num = env.action_space.n
    episode_durations = []

    model = ActorCritic(states_num, actions_num, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.alpha)
    steps_done = 0


    for i_episode in range(args.episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for t in count():
            state = args.encoder(state)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            steps_done += 1

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

            if done:
                episode_durations.append(t + 1)
                break

            state = next_state

        next_state = args.encoder(next_state)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks, args.gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy_coeff = max(args.entropy_end,
                            args.entropy_start *
                            (1 - steps_done / args.entropy_decay) + args.entropy_end * (
                            steps_done / args.entropy_decay))

        loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, int(sum(rewards))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of epochs to run')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Reward decay')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning Rate for training')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--entropy', type=float, default=0.01)
    parser.add_argument('--encoder', type=str, default='one_hot')
    parser.add_argument('--hidden-dim', type=int, default=50)
    parser.add_argument('--entropy-start', type=float, default=10)
    parser.add_argument('--entropy-end', type=float, default=0)
    parser.add_argument('--entropy-decay', type=int, default=1000000)

    args = parser.parse_args()
    if args.encoder == 'one_hot':
        args.encoder = one_hot
    elif args.encoder == 'complex_encoder':
        args.encoder = complex_encoder
    else:
        raise Exception('Please choose a valid encoder')
    start_time = time.time()
    train_taxi_pg(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

