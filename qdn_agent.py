import math
import random
from itertools import count

import matplotlib
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim
import matplotlib.pyplot as plt

from main import Environment, Feedback, State, is_valid_direction
from q_learn import q_state_11, game_stats

from qdn_replay_memory import ReplayMemory, Transition

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('matplotlib', matplotlib.__version__)
print('torch', torch.__version__, 'device', device)

is_ipython = 'inline' in matplotlib.get_backend()

actions = ['up', 'down', 'left', 'right']
ACTIONS_SIZE = len(actions)

reward_map = {
    Feedback.HIT_WALL: -10,
    Feedback.HIT_TAIL: -10,
    Feedback.ATE_FOOD: +10,
    Feedback.ELSE: 0,  # -0.1,
    Feedback.WOULD_180: -10,
}


class QDN11features(nn.Module):
    STATE_SIZE = 11
    REPLAY_SIZE = 10000

    def __init__(self,
                 action_size,
                 device=None,
                 hidden_size=[120, 120, 120]) -> None:
        super(QDN11features, self).__init__()

        # assert len(hidden_size) == 2, 'must be exactly 2 hidden layers'

        self.device = device
        self.h1 = nn.Linear(self.STATE_SIZE, hidden_size[0])
        self.drop1 = nn.Dropout(0.15)
        self.h2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.drop2 = nn.Dropout(0.15)
        self.h3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.drop3 = nn.Dropout(0.15)

        self.out = nn.Linear(hidden_size[1], action_size)

    def forward(self, input: Tensor) -> Tensor:
        x = F.relu(self.h1(input))
        x = self.drop1(x)
        x = F.relu(self.h2(x))
        x = self.drop2(x)
        x = F.relu(self.h3(x))
        x = self.drop3(x)
        return self.out(x)

    @staticmethod
    def get_state(grid, direction):
        state = q_state_11(grid, direction)
        # convert to torch, add batch dimension, to device
        return torch.FloatTensor(state) \
            .unsqueeze(0) \
            .to(device)


class QDNfullState(nn.Module):
    STATE_SIZE = Environment.rows * Environment.cols #+ 4
    REPLAY_SIZE = 50000
    INPUT_DIM = 5

    def __init__(self,
                 action_size,
                 device=None,
                 hidden_size=[120, 120, 120]) -> None:
        super(QDNfullState, self).__init__()

        # assert len(hidden_size) == 2, 'must be exactly 2 hidden layers'

        self.device = device
        self.conv1 = nn.Conv2d(self.INPUT_DIM, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=0):
            return (size - (kernel_size - 1) - 1) + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(Environment.rows)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(Environment.cols)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, action_size)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    @staticmethod
    def get_state(grid, direction):
        dir_up = dir_down = dir_left = dir_right = False
        if direction == 'up':
            dir_up = True
        elif direction == 'down':
            dir_down = True
        elif direction == 'left':
            dir_left = True
        elif direction == 'right':
            dir_right = True
        else:
            raise Exception('unknown direction ' + direction)
        state = torch.tensor(grid)
        # convert to torch, add batch dimension, to device
        hot = torch.nn.functional.one_hot(state, QDNfullState.INPUT_DIM)
        return hot \
            .reshape(QDNfullState.INPUT_DIM, hot.shape[0], hot.shape[1]) \
            .type(torch.float) \
            .unsqueeze(0) \
            .to(device)


class Agent(object):
    """Deep Q-learning agent."""
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.7
    EPS_END = 0.05
    EPS_DECAY = 2000

    steps_done = 0
    episode_durations = []

    def __init__(self,
                 action_space_size,
                 QDN):
        """Set parameters, initialize network."""

        self.action_space_size = action_space_size

        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QDN(action_space_size).to(device)
        print(self.policy_net)
        self.target_net = QDN(action_space_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(QDN.REPLAY_SIZE)

        self.experience_replay = None

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                net = self.policy_net(state)
                max_ = net.max(1)[1]
                return 'choice', max_.view(1, 1)
        else:
            return 'rnd', torch.tensor([[random.randrange(self.action_space_size)]], device=device, dtype=torch.long)

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def main():
    """
    Main method and execution start point
    """
    # set up matplotlib
    if is_ipython:
        from IPython import display

    plt.ion()

    ## Hyperparameter
    TARGET_UPDATE = 30
    QDN = QDNfullState
    num_episodes = 5000

    max_score = 0
    avg_len = np.array([0] * 100)

    env = Environment(False)
    direction = 'right'
    delay = 0

    agent = Agent(ACTIONS_SIZE, QDN)

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset_game()
        last_screen = QDN.get_state(env.grid, direction)
        current_screen = QDN.get_state(env.grid, direction)
        state = current_screen
        for t in count():
            if not env.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        flag = False
                    if event.type == pygame.KEYDOWN:
                        last_key_event = event
                        if event.key == pygame.K_q:
                            pygame.quit()
                            flag = False
                        if event.key == pygame.K_EQUALS:
                            delay += 10
                            print('delay', delay)
                        if event.key == pygame.K_MINUS:
                            delay -= 10
                            print('delay', delay)
                        if event.key == pygame.K_d:
                            DEBUG_OUT = not DEBUG_OUT

            # Select and perform an action
            reason, action = agent.select_action(state)
            new_direction = actions[action]
            if is_valid_direction(direction, new_direction):
                direction = new_direction

            feedback = env.step(direction)
            env.redrawWindow()
            reward = reward_map[feedback]
            if new_direction != direction: # would make an invalid action
                reward = reward_map[Feedback.WOULD_180]

            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = QDN.get_state(env.grid, direction)
            done = feedback in [Feedback.HIT_TAIL, Feedback.HIT_WALL]
            if not done:
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model()
            if done:
                max_score = max(env.snake_length(), max_score)
                game_stats(avg_len, env, i_episode, feedback, max_score, reason)
                agent.episode_durations.append(t + 1)
                # agent.plot_durations()
                break

            pygame.time.delay(delay)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    pygame.quit()


if __name__ == '__main__':
    main()
