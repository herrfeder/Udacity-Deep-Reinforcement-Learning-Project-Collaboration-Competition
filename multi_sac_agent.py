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
            self, state_size, action_size, random_seed, num_agents,
            hyperparameters={
                "buffer_size": 10000,
                "batch_size": 64,
                "lin_full_con_01": 256,
                "lin_full_con_02": 256,
                "gamma": 0.99,
                "tau": 5e-3,
                "learning_rate": 3e-4,
                "initial_rand_steps": 100,
                "policy_update": 2,
                "entropy_weight": 25e-5}
                ):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            Hyperparameters (dict): consists of the following parameters ->
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            lin_full_con_01 (int): Output Length first Fully Connected Layer
            lin_full_con_02 (int): Input Length second Fully Connected Layer
            gamma (float): discount factor
            tau (float): interpolation factor for soft update of target parameters
            learning_rate (float): learning rate for models
            initial_rand_steps (int): Number of steps the action is sampled from random dist
            policy_update (int): every n-Steps the Agent/Policy Network gets updated
            entropy_weight (int): Factor to influence the entropy added to actor loss
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
        self.initial_rand_steps = hyperparameters["initial_rand_steps"]
        self.policy_update = hyperparameters["policy_update"]
        self.static_alpha = hyperparameters["entropy_weight"]
        self.hyperparameters = hyperparameters
        self.num_agents = num_agents
        self.transition =[[]]*self.num_agents

        # Actor Network
        self.actor = Actor(state_size, action_size, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Q-Networks (Critics)
        # we have two Q Networks as it is a good mitigation for overestimation of Q-Networks
        self.critic_01 = CriticQ((self.state_size+self.action_size), seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_02 = CriticQ((self.state_size+self.action_size), seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.critic_01_optimizer = optim.Adam(self.critic_01.parameters(), lr=self.learning_rate)
        self.critic_02_optimizer = optim.Adam(self.critic_02.parameters(), lr=self.learning_rate)

        # Value Network (with target)
        self.value_local = Value(state_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.value_target = Value(state_size, seed=self.seed, fc1_units=self.lin_full_con_01, fc2_units=self.lin_full_con_02).to(device)
        self.hard_copy_weights(self.value_target, self.value_local)
        self.value_optimizer = optim.Adam(self.value_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(self.state_size, self.action_size, self.buffer_size, self.batch_size)
    
    def load_checkpoints(self):
        self.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        self.critic_01.load_state_dict(torch.load('checkpoint_critic_01.pth'))
        self.critic_02.load_state_dict(torch.load('checkpoint_critic_02.pth'))
        self.value_local.load_state_dict(torch.load('checkpoint_value_local.pth'))

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
        
        # OpenAI describes it here https://spinningup.openai.com/en/latest/algorithms/sac.html
        # as a "trick" to improve exploration in the beginning by sampling the selected action
        # from a uniform random distribution
        if step < self.initial_rand_steps:
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
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, s', a, r, done) tuples 
            step (int): Timestep over all episodes
        """
        state, next_state, action, reward, done = experiences
        # predict new action and output probability for this action
        new_action, log_prob = self.actor(state)

        # ----------- Training Q Function (update critics) ------------------- #
        critic_01_pred = self.critic_01(state, action)
        critic_02_pred = self.critic_02(state, action)
        value_target_pred = self.value_target(next_state)
        target_q_value = reward + self.gamma * value_target_pred * (1 - done)
        critic_01_loss = F.mse_loss(target_q_value.detach(), critic_01_pred)
        critic_02_loss = F.mse_loss(target_q_value.detach(), critic_02_pred)
        self.critic_01_optimizer.zero_grad() 
        critic_01_loss.backward()
        self.critic_01_optimizer.step()
        self.critic_02_optimizer.zero_grad() 
        critic_02_loss.backward()
        self.critic_02_optimizer.step() 
        # we extract the minimum from both Q-Networks to get the best prediction
        pred_new_q_value = torch.min(
            self.critic_01(state, new_action), self.critic_02(state, new_action)
        )

        # ----------------------- update value network ----------------------- #
        value_pred = self.value_local(state)
        value_target = pred_new_q_value - (self.static_alpha*log_prob)
        value_loss = F.mse_loss(value_pred, value_target.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # --------------- Training Policy Net (update actor) ----------------- #
        # update the policy network after every n "step"
        if step % self.policy_update == 0:   
            # Compute actor loss using Kullback-Leibler Divergence 
            difference_v_q = pred_new_q_value - value_pred.detach()
            actor_loss = (self.static_alpha*log_prob - difference_v_q).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.value_local, self.value_target)
            self.value_optimizer.step()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters using polyak averaging
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

    def __init__(self, state_size, action_size, memory_size, batch_size = 32):
        """Initialize a ReplayBuffer object.
        Params
        ======
            state_size (int): Dimension of Observation space
            action_size (int): Dimension of Action space
            memory_size (int): Length of Memory Buffer
            batch_size (int): size of each training batch (random selection)
        """

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.action_size = action_size
        self.state_size = state_size
        self.state_buf = np.zeros([self.memory_size, self.state_size], dtype=np.float32)
        self.next_state_buf = np.zeros([self.memory_size, self.state_size], dtype=np.float32)
        self.reward_buf = np.zeros([self.memory_size], dtype=np.float32)
        self.action_buf = np.zeros([self.memory_size, self.action_size], dtype=np.float32)
        self.done_buf = np.zeros(self.memory_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        idx = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        state = torch.FloatTensor(self.state_buf[idx]).to(device)
        next_state = torch.FloatTensor(self.next_state_buf[idx]).to(device)
        action = torch.FloatTensor(self.action_buf[idx].reshape(-1, self.action_size)).to(device)
        reward = torch.FloatTensor(self.reward_buf[idx].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(self.done_buf[idx].reshape(-1, 1)).to(device)

        return (state, next_state, action, reward, done)

    def __len__(self):
        """Return the current size of internal memory."""
        
        return self.size
