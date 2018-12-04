from model import DQN
import gym
from utils import *
from numpy import mean, std
import argparse


def eval_model(model, env, encoder, episodes=10, device='cpu'):
    model.eval()
    states_num = env.observation_space.n
    steps_done = []
    ep_reward_array = []

    for i in range(episodes):
        # Initialize the environment and state
        state = env.reset()
        state = encoder(state, states_num).to(device)
        ep_reward = 0
        done = False
        steps = 0

        while not done:
            # Select and perform an action
            steps += 1
            action = model(state.unsqueeze(0)).max(1)[1].view(1, 1)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            state = encoder(next_state, states_num).to(device)

        steps_done.append(steps)
        ep_reward_array.append(ep_reward)
    print('Evaluation done, average steps done = %.d, average accumulated reward = %d'
          % (int(mean(steps_done)),  int(mean(ep_reward_array))))
    model.train()
    return ep_reward_array


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    states_num = env.observation_space.n
    actions_num = env.action_space.n
    state = env.reset()
    state = args.encoder(state, states_num).to(device)
    states_dim = state.size()[0]

    # Create Model
    model = DQN(states_dim, args.hidden_dim, actions_num).to(device)
    model.load_state_dict(torch.load('current_model.pkl'))
    reward_array = eval_model(model, env, encoder=args.encoder, episodes=args.episodes, device=device)
    return reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of epochs to run')
    parser.add_argument('--encoder', type=str, default='one_hot')
    parser.add_argument('--hidden_dim', type=int, default=50)
    args = parser.parse_args()
    if args.encoder == 'one_hot':
        args.encoder = one_hot
    elif args.encoder == 'complex_encoder':
        args.encoder = complex_encoder
    else:
        raise Exception('Please choose a valid encoder')
    reward_array = test(args)
    eval_std = std(reward_array)
    eval_mean = mean(reward_array)
    print("Test done. Reward mean = %d, reward std = %d" % (eval_mean, eval_std))
