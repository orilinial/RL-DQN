from utils import moving_average
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == '__main__':
    acrobot_reward_eval = np.load('acrobot_reward_train.npy')
    plt.figure(1)
    # Accumulated reward plot
    plt.plot(range(len(acrobot_reward_eval)), acrobot_reward_eval)
    # On the same graph - rolling mean of accumulated reward
    plt.plot(range(len(acrobot_reward_eval)), moving_average(acrobot_reward_eval, periods=5))
    plt.title('Accumulated Reward Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.savefig('graphs/accumulated_reward_eval.png', bbox_inches='tight')
    plt.show()
    plt.close(1)
