from utils import *
from model_taxi import DQN
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import gym
from itertools import count
from eval_taxi_dqn import eval_model
import time
import argparse


def select_action_dqn(args, state, policy_net, steps_done, device):
    sample = random.random()
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) +
                        args.eps_end * (steps_done / args.eps_decay))
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

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state, dim=0)
    action_batch = torch.cat(batch.action, dim=0)
    reward_batch = torch.cat(batch.reward, dim=0)

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


def train_taxi_dqn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Taxi-v2')
    actions_num = env.action_space.n
    steps_done = 0
    episode_durations = []
    eval_reward_array = []

    # Create DQN models
    state = env.reset()
    state = args.encoder(state).to(device)
    states_dim = state.size()[1]
    policy_net = DQN(states_dim, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)
    target_net = DQN(states_dim, args.hidden_dim, actions_num, dropout_rate=args.dropout).to(device)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.alpha)

    # Create memory of available transitions
    memory = ReplayMemory(10000)

    # Main loop
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        state = env.reset()
        state = args.encoder(state).to(device)
        loss = None
        ep_reward = 0

        for t in count():
            # Select and perform an action
            action = select_action_dqn(args, state, policy_net, steps_done, device)
            next_state, reward, done, _ = env.step(action.item())
            steps_done += 1
            ep_reward += reward
            next_state = args.encoder(next_state).to(device)
            if done:
                next_state = None

            reward = torch.tensor([reward], device=device).unsqueeze(0)

            # Store the transition in memory
            if not (done and reward < 0):
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

    np.save('eval_reward_dqn_taxi.npy', np.array(eval_reward_array))
    torch.save(policy_net.state_dict(), 'dqn_taxi_model.pkl')
    print('Complete')
    env.render()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size to train on')
    parser.add_argument('--episodes', type=int, default=1000, help='Amount of train episodes to run')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma - discount factor')
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha - Learning rate')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--hidden-dim', type=int, default=50, help='Dimension of the hidden layer')
    parser.add_argument('--eps-start', type=float, default=1.0, help='Starting epsilon - in epsilon greedy method')
    parser.add_argument('--eps-end', type=float, default=0.1,
                        help='Final epsilon - in epsilon greedy method. When epsilon reaches this value it will stay')
    parser.add_argument('--eps-decay', type=int, default=50000,
                        help='Epsilon decay - how many steps until decaying to the final epsilon')
    parser.add_argument('--target-update', type=int, default=500, help='Number of steps until updating target network')
    parser.add_argument('--reg-param', type=float, default=0, help='L1 regulatization parameter')
    parser.add_argument('--encoder', type=str, default='one_hot',
                        help='Which encoder to choose, one_hot, or complex_encoder')
    args = parser.parse_args()
    if args.encoder == 'one_hot':
        args.encoder = one_hot
    elif args.encoder == 'complex_encoder':
        args.encoder = complex_encoder
    else:
        raise Exception('Please choose a valid encoder')
    start_time = time.time()
    train_taxi_dqn(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

