import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, CriticQ, Value

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(
            self, state_size, action_size, random_seed,
            hyperparameters={
                "buffer_size": int(1e5),
                "batch_size": 64,
                "lin_full_con_01": 256,
                "lin_full_con_02": 256,
                "gamma": 0.99,
                "tau": 5e-3,
                "learning_rate": 3e-4,
            ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            lin_full_con_01 (int): Output Length first Fully Connected Layer
            lin_full_con_02 (int): Input Length second Fully Connected Layer
            gamma (float): discount factor
            tau (float): interpolation factor for soft update of target parameters
            learning_rate (float): learning rate for models
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.buffer_size = hyperparameters["buffer_size"]
        self.batch_size = hyperparameters["batch_size"]
        self.gamma = hyperparameters["gamma"]
        self.tau = hyperparameters["tau"]
        self.lin_full_con_01 = hyperparameters["lin_full_con_01"]
        self.lin_full_con_02 = hyperparameters["lin_full_con_02"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.weight_decay = hyperparameters["weight_decay"]
        self.noise_scalar = hyperparameters["noise_scalar"]
        self.hyperparameters = hyperparameters
        self.policy_update = 2
        self.initial_random_steps = 100
        self.num_agents = 2
        self.transition =[[]]*self.num_agents

        # Actor Network
        self.actor = Actor(state_size, action_size, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Critics (One per agent)
        self.critic_01 = CriticQ((self.state_size+self.action_size), seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_02 = CriticQ((self.state_size+self.action_size), seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_01_optimizer = optim.Adam(self.critic_01.parameters(), lr=self.learning_rate)
        self.critic_02_optimizer = optim.Adam(self.critic_02.parameters(), lr=self.learning_rate)

        # Value Network (with target)
        self.value_local = Value(state_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.value_target = Value(state_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.hard_copy_weights(self.value_target, self.value_local)
        self.value_optimizer = optim.Adam(self.value_local.parameters(), lr=self.learning_rate)

        # Logarithmic Tensor
        self.target_alpha = -np.prod((self.action_size,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(self.state_size, self.action_size, self.buffer_size, self.batch_size)
    
    def load_checkpoints(self):
        self.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        self.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(self.num_agents):
            self.transition[i] += [reward[i], next_state[i], done[i]]
            self.memory.add(*self.transition[i])

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, timestep)

    def act(self, state, step):
        """Returns actions for given state as per current policy."""
        if step < self.initial_random_steps:
            selected_action = np.random.uniform(-1, 1, (self.num_agents, self.action_size))
        else:
            selected_action = []
            for i in range(self.num_agents):
                action = self.actor(
                    torch.FloatTensor(state[i]).to(device)
                )[0].detach().cpu().numpy()
                selected_action.append(action)
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)

        for i in range(self.num_agents):
            self.transition[i] = [state[i], selected_action[i]]
        
        return selected_action


    def learn(self, experiences, step):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state, next_state, action, reward, done = experiences
        new_action, log_prob = self.actor(state)
        # ---------------------------- update probability function ------------- #
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()
        ).mean()
        self.log_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_optimizer.step()
        # ---------------------------- calculate losses ------------------------ #
        alpha = self.log_alpha.exp()
        mask = 1 - done
        critic_01_pred = self.critic_01(state, action)
        critic_02_pred = self.critic_02(state, action)
        value_target_pred = self.value_target(next_state)
        q_target = reward + self.gamma * value_target_pred * mask
        critic_01_loss = F.mse_loss(q_target.detach(), critic_01_pred)
        critic_02_loss = F.mse_loss(q_target.detach(), critic_02_pred)

        value_pred = self.value_local(state)
        critic_pred = torch.min(
            self.critic_01(state, new_action), self.critic_02(state, new_action)
        )
        value_target = critic_pred - alpha * log_prob
        value_loss = F.mse_loss(value_pred, value_target.detach())
        # ---------------------------- update actor ---------------------------- #
        if step % self.policy_update == 0:   
            # Compute actor loss
            advantage = critic_pred - value_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.value_local, self.value_target)
        else:
            actor_loss = torch.zeros(1)

        # ---------------------------- update critics -------------------------- #
        self.critic_01_optimizer.zero_grad() 
        critic_01_loss.backward()
        self.critic_01_optimizer.step()
        self.critic_02_optimizer.zero_grad() 
        critic_02_loss.backward()
        self.critic_02_optimizer.step()

        critic_loss = critic_01_loss + critic_02_loss
        # ----------------------- update value network ----------------------- #
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.action_size = action_dim
        self.ptr, self.size = 0, 0


    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a new experience to memory."""
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idx = np.random.choice(self.size, size=self.batch_size, replace=False)
        state = torch.FloatTensor(self.obs_buf[idx]).to(device)
        next_state = torch.FloatTensor(self.next_obs_buf[idx]).to(device)
        action = torch.FloatTensor(self.acts_buf[idx].reshape(-1, self.action_size)).to(device)
        reward = torch.FloatTensor(self.rews_buf[idx].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(self.done_buf[idx].reshape(-1, 1)).to(device)

        return (state, next_state, action, reward, done)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size
