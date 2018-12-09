import gym
import argparse
import time
from PIL import Image
import torchvision.transforms as T
from utils import *
from itertools import count
from model_acrobot import DQN
import torch.optim as optim
import torch.nn.functional as F


transforms = T.Compose([T.ToPILImage(),
                        T.Grayscale(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

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
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def get_screen(env):
    # transpose into torch order (CHW)
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.uint8)
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    screen = 1 - transforms(screen).unsqueeze(0).to(device)
    return screen


def optimize_model(args, policy_net, target_net, optimizer, memory):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
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
    env = gym.make('Acrobot-v1').unwrapped
    env.reset()

    steps_done = 0
    episode_durations = []

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    for i_episode in range(args.episodes):
        # Initialize the environment and state
        env.reset()
        ep_reward = 0
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(args, state, policy_net, steps_done)
            steps_done += 1
            _, reward, done, _ = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model(args, policy_net, target_net, optimizer, memory)

            if done:
                episode_durations.append(t + 1)
                break

        # Update the target network
        if steps_done % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print("Episode %d complete, episode duration = %d, loss = %.3f, reward = %d" %
                  (i_episode, episode_durations[-1], loss, ep_reward))
    print('Complete')
    env.render()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size to train on')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--states-dim', type=int, default=40)
    parser.add_argument('--hidden-dim', type=int, default=50)
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.1)
    parser.add_argument('--eps-decay', type=int, default=50000)
    parser.add_argument('--target-update', type=int, default=500,
                        help='Number of steps until updating target network')

    args = parser.parse_args()

    start_time = time.time()
    train_acrobot(args)
    print('Run finished successfully in %s seconds' % round(time.time() - start_time))

