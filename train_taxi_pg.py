from utils import *
from model import Policy
import numpy as np
import torch.optim as optim
import torch
import gym
from itertools import count
from eval_taxi import eval_model
import matplotlib
from torch.distributions import Categorical
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import argparse


def select_action_pg(state, policy_net):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    action_vec = policy_net(state)
    c = Categorical(action_vec)
    action = c.sample()
    # Add log probability of our chosen action to our history
    policy_net.policy_history = torch.cat([policy_net.policy_history, c.log_prob(action)])
    entropy = c.entropy()
    return action.item(), entropy.mean()


def update_policy(args, policy_net, optimizer, reward_episode, ep_entropy, entropy_coeff):
    reward = 0
    reward_array = []

    # Discount future rewards back to the present using gamma
    for r in reward_episode[::-1]:
        reward = r + args.gamma * reward
        reward_array.insert(0, reward)

    # Scale rewards
    reward_array = torch.FloatTensor(reward_array)
    reward_array = (reward_array - reward_array.mean()) / (reward_array.std() + float(np.finfo(np.float32).eps))

    # Calculate loss
    loss = torch.sum(torch.mul(policy_net.policy_history, Variable(reward_array)).mul(-1), -1)
    loss -= entropy_coeff * ep_entropy

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy_net.policy_history = Variable(torch.Tensor())

    return loss.item()

def train_taxi_pg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    env = env.unwrapped
    actions_num = env.action_space.n
    episode_durations = []
    total_reward = []

    state = env.reset()
    state = args.encoder(state).to(device)
    states_dim = state.size()[1]
    steps_done = 0

    policy_net = Policy(states_dim, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.alpha)
    for i_episode in range(args.episodes):
        state = env.reset()  # Reset environment and record the starting state
        state = args.encoder(state).to(device)
        ep_reward = []
        ep_entropy = 0

        for t in count():
            action, entropy = select_action_pg(state, policy_net)
            ep_entropy += entropy
            # Step through environment using chosen action
            next_state, reward, done, _ = env.step(action)
            steps_done += 1

            # Save reward
            ep_reward.append(reward)
            if done:
                episode_durations.append(t + 1)
                break
            next_state = args.encoder(next_state)
            state = next_state

        entropy_coeff = max(args.entropy_end,
                            args.entropy_start *
                            (1 - steps_done / args.entropy_decay) + args.entropy_end * (
                            steps_done / args.entropy_decay))

        loss = update_policy(args, policy_net, optimizer, ep_reward, ep_entropy, entropy_coeff)
        total_ep_reward = int(sum(ep_reward))
        total_reward.append(total_ep_reward)

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d entropy coeff = %.3f" %
              (i_episode, episode_durations[-1], loss, total_ep_reward, entropy_coeff))

    np.save('pg_reward_array.npy', np.array(total_reward))
    torch.save(policy_net.state_dict(), 'pg_taxi_model.pkl')
    print('Complete')
    env.render()
    env.close()

    # Creating plots:
    plt.figure(1)
    # Accumulated reward plot
    plt.plot(range(len(total_reward)), total_reward)
    # On the same graph - rolling mean of accumulated reward
    plt.plot(range(len(total_reward)), moving_average(total_reward))
    plt.title('Accumulated Reward Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.savefig('graphs/accumulated_reward_pg.png', bbox_inches='tight')
    plt.close(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--entropy-start', type=float, default=10)
    parser.add_argument('--entropy-end', type=float, default=0)
    parser.add_argument('--entropy-decay', type=int, default=1000000)
    parser.add_argument('--encoder', type=str, default='one_hot')
    parser.add_argument('--hidden-dim', type=int, default=50)
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

