from model_taxi import DQN
import gym
from utils import one_hot, complex_encoder
from numpy import mean, std
import argparse
import torch


def eval_model(model, env, encoder, episodes=10, device='cpu'):
    model.eval()
    steps_done = []
    ep_reward_array = []

    for i in range(episodes):
        # Initialize the environment and state
        state = env.reset()
        state = encoder(state).to(device)
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            # Select and perform an action
            steps += 1
            action = model(state).max(1)[1].view(1, 1)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            state = encoder(next_state).to(device)

        steps_done.append(steps)
        ep_reward_array.append(ep_reward)

    print('Evaluation done, average steps done = %.1f, average accumulated reward = %.1f, std = %.1f'
          % (float(mean(steps_done)),  float(mean(ep_reward_array)), float(std(ep_reward_array))))
    model.train()
    return ep_reward_array


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    actions_num = env.action_space.n
    state = env.reset()
    state = args.encoder(state).to(device)
    states_dim = state.size()[1]

    # Create Model
    model = DQN(states_dim, args.hidden_dim, actions_num).to(device)
    model.load_state_dict(torch.load('dqn_taxi_model.pkl'))
    eval_model(model, env, encoder=args.encoder, episodes=args.episodes, device=device)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of epochs to run')
    parser.add_argument('--encoder', type=str, default='one_hot')
    parser.add_argument('--hidden-dim', type=int, default=50)
    args = parser.parse_args()

    if args.encoder == 'one_hot':
        args.encoder = one_hot
    elif args.encoder == 'complex_encoder':
        args.encoder = complex_encoder
    else:
        raise Exception('Please choose a valid encoder')

    print("Starting test on TAXI environment, with DQN method.")
    test(args)
    print("Test done.")
