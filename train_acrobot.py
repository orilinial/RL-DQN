from utils import *
from itertools import count
from model_acrobot import DQN
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
from utils import get_screen
import gym
from eval_acrobot import eval_model


def select_action(args, state, policy_net, steps_done, device):
    sample = random.random()
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) + args.eps_end * (
                        steps_done / args.eps_decay))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(0, 3)]], device=device, dtype=torch.long)


def optimize_model(args, policy_net, target_net, optimizer, memory, success_memory, steps_done, device):
    if len(memory) < args.batch_size:
        return

    # Sample batch from memory or success memory
    sample = random.random()
    success_threshold = max(args.success_ratio_end,
                            args.success_ratio_start * (1 - steps_done / args.success_ratio_decay) +
                            args.success_ratio_end * (steps_done / args.success_ratio_decay))

    if (sample > success_threshold) or (len(success_memory) < args.batch_size):
        transitions = memory.sample(args.batch_size)
    else:
        transitions = success_memory.sample(args.batch_size)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(args.batch_size, device=device)
    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        if args.double_dqn:
            next_actions = policy_net(non_final_next_states).max(1, keepdim=True)[1]
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
        else:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def train_acrobot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Acrobot-v1')
    env.reset()

    steps_done = 0
    episode_durations = []
    reward_array = []
    eval_arr = []

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    mem_size = 200000
    memory = ReplayMemory(mem_size)
    success_memory = ReplayMemory(round(mem_size*0.25))
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        env.reset()
        ep_reward = 0
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(args, state, policy_net, steps_done, device)
            steps_done += 1
            _, reward, done, _ = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            if not (done and reward < 0):
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Push to success memory
            if done and reward.item() == 0:
                ep_range = min(t+1, 200)
                for i in range(ep_range):
                    if memory.position-i-1 < 0:
                        break
                    c_state, c_action, c_next_state, c_reward = memory.memory[memory.position-1-i]
                    success_memory.push(c_state, c_action, c_next_state, c_reward)

            # Perform one step of the optimization (on the target network)
            loss = optimize_model(args, policy_net, target_net, optimizer, memory, success_memory, steps_done, device)

            # Update the target network
            if steps_done % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # episode_stop = 999 if i_episode < 150 else 499
            if done:
                episode_durations.append(t + 1)
                break

        reward_array.append(ep_reward)

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, ep_reward))

        if i_episode % 20 == 0 and i_episode != 0:
            eval_res = eval_model(policy_net, env, episodes=10, device=device)
            eval_arr.append(np.mean(eval_res))

    np.save('acrobot_reward_eval.npy', np.array(eval_arr))
    np.save('acrobot_reward_train.npy', np.array(reward_array))
    torch.save(policy_net.state_dict(), 'acrobot_model.pkl')
    print('Complete')
    env.render()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to train on')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--states-dim', type=int, default=40)
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.2)
    parser.add_argument('--eps-decay', type=int, default=1000000)
    parser.add_argument('--target-update', type=int, default=500,
                        help='Number of steps until updating target network')
    parser.add_argument('--success-ratio-start', type=float, default=0.5)
    parser.add_argument('--success-ratio-end', type=float, default=0.2)
    parser.add_argument('--success-ratio-decay', type=int, default=1000000)
    parser.add_argument('--double-dqn', type=bool, default=True)

    args = parser.parse_args()

    start_time = time.time()
    train_acrobot(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

