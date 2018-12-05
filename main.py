from utils import *
import argparse
from model import DQN
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import gym
import random
from itertools import count
from eval_model import eval_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def select_action(args, state, policy_net, steps_done, device):
    sample = random.random()
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) + args.eps_end * (steps_done / args.eps_decay))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)


def optimize_model(args, policy_net, target_net, optimizer, memory, device):
    # First check if there is an availalbe batch
    if len(memory) < args.batch_size:
        return 0

    # Sample batch to learn from
    transitions = memory.sample(args.batch_size)

    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

    """
    state_batch = None
    action_batch = None
    next_state_batch = None
    reward_batch = None
    non_final_mask = None
    for i, transition in enumerate(transitions):
        if i == 0:
            state_batch = transition.state.unsqueeze(0)
            action_batch = transition.action.unsqueeze(0)
            reward_batch = transition.reward.unsqueeze(0)
            if transition.next_state is not None:
                next_state_batch = transition.next_state.unsqueeze(0)
                non_final_mask = torch.tensor([[True]])
            else:
                non_final_mask = torch.tensor([[False]])
        else:
            state_batch = torch.cat((state_batch, transition.state.unsqueeze(0)), 0)
            action_batch = torch.cat((action_batch, transition.action.unsqueeze(0)), 0)
            reward_batch = torch.cat((reward_batch, transition.reward.unsqueeze(0)), 0)

            if transition.next_state is not None:
                non_final_mask = torch.cat((non_final_mask, torch.tensor([[True]])))
                if next_state_batch is None:
                    next_state_batch = transition.next_state.unsqueeze(0)
                else:
                    next_state_batch = torch.cat((next_state_batch, transition.next_state.unsqueeze(0)), 0)
            else:
                non_final_mask = torch.cat((non_final_mask, torch.tensor([[False]])))
    """

    # Compute Q(s_t, a) - the model computes Q(s_t),
    # Then, using gather, we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros((args.batch_size, 1), device=device)
    # next_state_values[non_final_mask] = target_net(next_state_batch).max(1)[0].detach()
    # print(target_net(non_final_next_states).max(1)[0].detach())
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].unsqueeze(1).detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch.float()

    # Compute Huber loss
    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
    if args.reg_param > 0:
        regularization = 0.0
        for param in policy_net.parameters():
            regularization += torch.sum(torch.abs(param))
        loss += args.reg_param * regularization

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    states_num = env.observation_space.n
    actions_num = env.action_space.n
    steps_done = 0
    episode_durations = []
    eval_reward_array = []

    # Create DQN models
    state = env.reset()
    state = args.encoder(state, states_num).to(device)
    states_dim = state.size()[1]
    policy_net = DQN(states_dim, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)
    target_net = DQN(states_dim, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)

    print("Net created. number of params: {}".format(sum(param.numel() for param in policy_net.parameters())))

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.alpha)

    # Create memory of available transitions
    memory = ReplayMemory(10000)

    # Main loop
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        state = env.reset()
        state = args.encoder(state, states_num).to(device)
        loss = None
        ep_reward = 0

        for t in count():
            # Select and perform an action
            action = select_action(args, state, policy_net, steps_done, device)
            next_state, reward, done, _ = env.step(action.item())
            steps_done += 1
            ep_reward += reward
            next_state = args.encoder(next_state, states_num).to(device)
            if done:
                next_state = None

            reward = torch.tensor([reward], device=device).unsqueeze(0)

            # Store the transition in memory
            if not(done and reward < 0):
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model(args, policy_net, target_net, optimizer, memory, device)
            # Update the target network
            if steps_done % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                episode_durations.append(t + 1)
                break
        if i_episode % 10 == 0 and i_episode != 0:
            print("Evaluation:")
            eval_reward = eval_model(model=policy_net, env=env, encoder=args.encoder, episodes=10, device=device)
            eval_mean_reward = np.mean(eval_reward)
            eval_reward_array.append(eval_mean_reward)
        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, ep_reward))
    
    np.save('eval_reward_array.npy', np.array(eval_reward_array))
    torch.save(policy_net.state_dict(), 'current_model.pkl')
    print('Complete')
    env.render()
    env.close()

    # Creating plots:
    plt.figure(1)
    # Accumulated reward plot
    plt.plot(range(len(eval_reward_array)), eval_reward_array)
    # On the same graph - rolling mean of accumulated reward
    plt.plot(range(len(eval_reward_array)), moving_average(eval_reward_array))
    plt.title('Accumulated Reward Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    plt.savefig('graphs/accumulated_reward.png', bbox_inches='tight')
    plt.close(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size to train on')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of epochs to run')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='Reward decay')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Learning Rate for training')
    parser.add_argument('--target-update', type=int, default=500,
                        help='Number of steps until updating target network')
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.1)
    parser.add_argument('--eps-decay', type=int, default=50000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--reg-param', type=float, default=0)
    parser.add_argument('--encoder', type=str, default='one_hot')
    parser.add_argument('--hidden-dim', type=int, default=50)
    args = parser.parse_args()
    if args.encoder == 'one_hot':
        args.encoder = one_hot
    elif args.encoder == 'complex_encoder':
        args.encoder = complex_encoder
    else:
        raise Exception('Please choose a valid encoder')
    main(args)
