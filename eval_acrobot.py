from model_acrobot import DQN
import gym
from numpy import mean, std
import argparse
import torch
from utils import get_screen


def eval_model(model, env, episodes=1, device='cpu'):
    model.eval()
    ep_duration = []
    ep_reward_array = []

    for i_episode in range(episodes):
        # Initialize the environment and state
        env.reset()
        ep_reward = 0
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        done = False
        steps = 0

        while not done:
            # Select and perform an action
            action = model(state).max(1)[1].view(1, 1)
            steps += 1
            _, reward, done, _ = env.step(action.item()-1)
            ep_reward += reward

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)

            next_state = current_screen - last_screen

            # Move to the next state
            state = next_state

        ep_duration.append(steps)
        ep_reward_array.append(ep_reward)

    print('Evaluation done, average steps done = %.1f, average accumulated reward = %.1f, std = %.1f'
          % (float(mean(ep_duration)),  float(mean(ep_reward_array)), float(std(ep_reward_array))))
    model.train()
    return ep_reward_array


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Acrobot-v1')

    # Create Model
    model = DQN().to(device)
    model.load_state_dict(torch.load(args.model))
    eval_model(model, env, episodes=args.episodes)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of epochs to run')
    parser.add_argument('--model', type=str, default='acrobot_model.pkl')
    args = parser.parse_args()

    print("Starting test on ACROBOT environment, with DQN method.")
    test(args)
    print("Test done.")
