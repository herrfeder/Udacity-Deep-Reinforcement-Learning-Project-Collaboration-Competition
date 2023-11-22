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
                "batch_size": 256,
                "lin_full_con_01": 128,
                "lin_full_con_02": 128,
                "gamma": 0.99,
                "tau": 1e-3,
                "lr_actor": 3e-4,
                "lr_critic": 3e-4,
                "lr_value": 3e-4,
                "weight_decay": 0,
                "noise_scalar": 0.25}
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
            lr_actor (float): learning rate of actor
            lr_critic (float): learning rate of critic
            lr_value (float): learning rate of value
            weight_decay (int): L2 weight decay
            noise_scalar (float): Constant noise added to choosen action
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
        self.lr_actor = hyperparameters["lr_actor"]
        self.lr_critic = hyperparameters["lr_critic"]
        self.lr_value = hyperparameters["lr_value"]
        self.weight_decay = hyperparameters["weight_decay"]
        self.noise_scalar = hyperparameters["noise_scalar"]
        self.hyperparameters = hyperparameters

        self.initial_random_steps = 10
        self.num_agents = 2
        self.transition =[[]]*self.num_agents
        print(self.transition)

        # Actor Network
        self.actor = Actor(state_size, action_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critics (One per agent)
        self.critic_01 = CriticQ(state_size, action_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_02 = CriticQ(state_size, action_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_01_optimizer = optim.Adam(self.critic_01.parameters(), lr=self.lr_critic)
        self.critic_02_optimizer = optim.Adam(self.critic_02.parameters(), lr=self.lr_critic)

        # Value Network (with target)
        self.value_local = Value(state_size, action_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.value_target = Value(state_size, action_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.hard_copy_weights(self.value_target, self.value_local)
        self.value_optimizer = optim.Adam(self.value_local.parameters(), lr=self.lr_value)

        # Logarithmic Tensor
        self.target_alpha = -np.prod((self.action_size,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
    
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
            self.learn(experiences)

    def act(self, state, step):
        """Returns actions for given state as per current policy."""
        #state = torch.from_numpy(state).float().to(device)
        #self.actor.eval()
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

    def reset(self):
        pass

    def learn(self, experiences):
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
        states, actions, rewards, next_states, dones = experiences
        new_action, log_prob = self.actor(states)
        # ---------------------------- update probability function ------------- #
        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()
        ).mean()
        self.log_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_optimizer.step()
        # ---------------------------- calculate losses ------------------------ #
        alpha = self.log_alpha.exp()
        mask = 1 - dones
        critic_01_pred = self.critic_01(states, actions)
        critic_02_pred = self.critic_02(states, actions)
        value_target = self.value_target(next_states)
        q_target = reward + self.gamma * value_target * mask
        critic_01_loss = F.mse_loss(q_target.detach(), critic_01_pred)
        critic_02_loss = F.mse_loss(q_target.detach(), critic_02_pred)

        value_pred = self.value_local(states)
        critic_pred = torch.min(
            self.critic_01(state, new_action), self.critic_02(state, new_action)
        )
        value_target = critic_pred - alpha * log_prob
        value_loss = F.mse_loss(value_pred, value_target.detach())
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        advantage = q_pred - v_pred.detach()
        actor_loss = (alpha * log_prob - advantage).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ---------------------------- update critics -------------------------- #
        self.critic_01_optimizer.zero_grad() 
        critc_01_loss.backward()
        self.critic_01_optimizer.step()
        self.critic_02_optimizer.zero_grad() 
        critc_02_loss.backward()
        self.critic_02_optimizer.step()

        critic_loss = critic_01_loss + crictic_02_loss
        # ----------------------- update value network ----------------------- #
        self.soft_update(self.value_local, self.value_target)
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

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
