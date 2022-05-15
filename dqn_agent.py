# Defines a DQN agent: how it selects actions, learns, etc.
# Borrows heavily from the Udacity Deep Reinforcement Learning course

import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork


BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.95
LR = 5e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, update_every=1, ddqn=False, ddqn_mean=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            update_every: how often to update the network
            ddqn: True for double DQN
            ddqn_mean: If ddqn is True, then this is whether to act using mean(Q1, Q2); the alternative uses random_choice(Q1, Q2)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.update_every = update_every
        self.ddqn = ddqn
        self.ddqn_mean = ddqn_mean # For Double DQN: True to use mean of Q1 and Q2; False to use random one of Q1 and Q2

        # Q-Network
        if not self.ddqn:
            print("Creating new DQN agent with gamma =", GAMMA)
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        else:
            if self.ddqn_mean:
                print("Creating new Double DQN agent that moves using mean(Q1, Q2) with gamma =", GAMMA)
            else:
                print("Creating new Double DQN agent that moves using random_choice(Q1, Q2) with gamma =", GAMMA)
            self.qnetwork1 = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork2 = QNetwork(state_size, action_size, seed + 1000).to(device)
            self.optimizer1 = optim.Adam(self.qnetwork1.parameters(), lr=LR)
            self.optimizer2 = optim.Adam(self.qnetwork2.parameters(), lr=LR)

        # Replay memory
        print("Initializing replay buffer with buffer size", BUFFER_SIZE, "and batch size", BATCH_SIZE)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, seed)
        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if not self.ddqn:
                self.qnetwork_local.eval()
                action_values = self.qnetwork_local(state)
                self.qnetwork_local.train()
            else:
                self.qnetwork1.eval()
                self.qnetwork2.eval()
                action_values1 = self.qnetwork1(state)
                action_values2 = self.qnetwork2(state)
                self.qnetwork1.train()
                self.qnetwork2.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            if not self.ddqn:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                if self.ddqn_mean:
                    action_values = [(i + j) / 2.0 for (i, j) in zip(action_values1.cpu().data.numpy(), action_values2.cpu().data.numpy())]
                else:
                    if random.random() < 0.5:
                        action_values = action_values1.cpu().data.numpy()
                    else:
                        action_values = action_values2.cpu().data.numpy()
                return np.argmax(action_values)

        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if not self.ddqn:
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)


        else:

            if random.random() < 0.5:
                # 1 learns from 2
                learn12 = True

                # Get max predicted Q values (for next states) from target model
                argmax_actions = torch.argmax(self.qnetwork1(next_states), 1).unsqueeze(1)
                Q_targets_next = self.qnetwork2(next_states).gather(1, argmax_actions)

                # Compute Q targets for current states 
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

                # Get expected Q values from local model
                Q_expected = self.qnetwork1(states).gather(1, actions)
            else:
                # 2 learns from 1
                learn12 = False

                # Get max predicted Q values (for next states) from target model
                argmax_actions = torch.argmax(self.qnetwork2(next_states), 1).unsqueeze(1)
                Q_targets_next = self.qnetwork1(next_states).gather(1, argmax_actions)

                # Compute Q targets for current states 
                Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

                # Get expected Q values from local model
                Q_expected = self.qnetwork2(states).gather(1, actions)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        if not self.ddqn:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            if learn12:
                self.optimizer1.zero_grad()
                loss.backward()
                self.optimizer1.step()
            else:
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer2.step()


    def hard_update(self):
        if self.ddqn:
            return
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data)              


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
