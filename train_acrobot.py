from utils import *
from itertools import count
from model_acrobot import DQN
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import time
import argparse
from utils import get_screen
import gym
from eval_acrobot import eval_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(args, state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = max(args.eps_end,
                        args.eps_start * (1 - steps_done / args.eps_decay) + args.eps_end * (
                        steps_done / args.eps_decay))
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(0, 3)]], device=device, dtype=torch.long)


def optimize_model(args, policy_net, target_net, optimizer, memory, success_memory):
    mem_batch_size = round(args.batch_size * 0.8)
    succ_batch_size = args.batch_size - mem_batch_size
    if len(memory) < args.batch_size:
        return

    if len(success_memory) < succ_batch_size:
        transitions = memory.sample(args.batch_size)
    else:
        mem_transitions = memory.sample(mem_batch_size)
        succ_transitions = success_memory.sample(succ_batch_size)
        transitions = mem_transitions + succ_transitions

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

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
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
    memory = ReplayMemory(10000)
    success_memory = ReplayMemory(10000)
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        env.reset()
        ep_reward = 0
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(args, state, policy_net, steps_done)
            steps_done += 1
            _, reward, done, _ = env.step(action.item()-1)
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
                ep_range = min(t+1, 100)
                for i in range(ep_range):
                    if memory.position - i < 0:
                        break
                    try:
                        c_state, c_action, c_next_state, c_reward = memory.memory[memory.position-i]
                    except:
                        print('Error occurred')
                        print('memory.position = ' + str(memory.position))
                        print('i = ' + str(i))
                        raise
                    success_memory.push(c_state, c_action, c_next_state, c_reward)

            # Perform one step of the optimization (on the target network)
            loss = optimize_model(args, policy_net, target_net, optimizer, memory, success_memory)

            if done:
                episode_durations.append(t + 1)
                break

        reward_array.append(ep_reward)

        if i_episode % 20 == 0 and i_episode != 0:
            eval_res = eval_model(policy_net, env, episodes=1, device=device)
            eval_arr.append(eval_res)

        # Update the target network
        if steps_done % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
              (i_episode, episode_durations[-1], loss, ep_reward))

    np.save('eval_arr.npy', np.array(eval_arr))
    np.save('acrobot_reward_array.npy', np.array(reward_array))
    torch.save(policy_net.state_dict(), 'acrobot_model.pkl')
    print('Complete')
    env.render()
    env.close()

    # Creating plots:
    if args.plot:
        plt.figure(1)
        # Accumulated reward plot
        plt.plot(range(len(reward_array)), reward_array)
        # On the same graph - rolling mean of accumulated reward
        plt.plot(range(len(reward_array)), moving_average(reward_array))
        plt.title('Accumulated Reward Per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Accumulated Reward')
        plt.savefig('graphs/accumulated_reward_dqn.png', bbox_inches='tight')
        plt.close(1)


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
    parser.add_argument('--eps-end', type=float, default=0.1)
    parser.add_argument('--eps-decay', type=int, default=50000)
    parser.add_argument('--target-update', type=int, default=1500,
                        help='Number of steps until updating target network')
    parser.add_argument('--plot', type=bool, default=True)

    args = parser.parse_args()

    start_time = time.time()
    train_acrobot(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

