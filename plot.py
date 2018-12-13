from utils import moving_average
import numpy as np
import matplotlib
import argparse
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plot(args):
    acrobot_reward_eval = np.load(args.path)
    plt.figure(1)
    # Accumulated reward plot
    plt.plot(range(len(acrobot_reward_eval)), acrobot_reward_eval)
    # On the same graph - rolling mean of accumulated reward
    plt.plot(range(len(acrobot_reward_eval)), moving_average(acrobot_reward_eval, periods=5))
    plt.title('Accumulated Reward Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    if args.save_fig:
        plt.savefig('accumulated_reward.png', bbox_inches='tight')
    if args.show_fig:
        plt.show()
    plt.close(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path', type=str, default='eval_reward_dqn_taxi.npy', help='Path of the numpy array to plot')
    parser.add_argument('--save-fig', type=bool, default=False)
    parser.add_argument('--show-fig', type=bool, default=True)
    args = parser.parse_args()
    plot(args)
