from model_taxi import Policy
import gym
from utils import one_hot
from numpy import mean, std
import argparse
import torch
from torch.distributions import Categorical


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    actions_num = env.action_space.n
    state = env.reset()
    state = one_hot(state).to(device)
    states_dim = state.size()[1]

    # Create Model
    model = Policy(states_dim, args.hidden_dim, actions_num).to(device)
    model.load_state_dict(torch.load('pg_taxi_model.pkl'))
    model.eval()
    steps_done = []
    ep_reward_array = []

    for i in range(args.episodes):
        # Initialize the environment and state
        state = env.reset()
        state = one_hot(state).to(device)
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            # Select and perform an action
            steps += 1
            action_vec = model(state)
            c = Categorical(action_vec)
            action = c.sample()
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            state = one_hot(next_state).to(device)

        steps_done.append(steps)
        ep_reward_array.append(ep_reward)

    print('Evaluation done, average steps done = %.1f, average accumulated reward = %.1f, std = %.1f'
          % (float(mean(steps_done)),  float(mean(ep_reward_array)), float(std(ep_reward_array))))
    model.train()
    return ep_reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of epochs to run')
    parser.add_argument('--hidden-dim', type=int, default=50)
    parser.add_argument('--model', type=str, default='dqn',
                        help='Which model to use (pg or dqn)')
    args = parser.parse_args()
    print("Starting test on TAXI environment, with Policy Gradient method.")
    test(args)
    print("Test done.")
