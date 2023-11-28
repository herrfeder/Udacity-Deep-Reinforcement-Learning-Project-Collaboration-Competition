import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, 
                 seed=0, epsilon=1e-7, fc1_units=256,
                 fc2_units=256, log_std_min=-20, log_std_max=2
                ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): seed for initializing pseudo number generators
            epsilon (float): Epsilon Decay factor
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.epsilon = epsilon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.mean_layer = nn.Linear(fc2_units, action_size)
        self.log_std_layer = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mean_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.mean_layer.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std_layer.bias.data.uniform_(-3e-3, 3e-3)

    def sample_normal(self, state):
        """Instead of returning results of activation in SAC the Policy Network
        will sample the resulting action from a normal distribution.
        The Result will be a action and a log probability for this"""

        mean = self.mean_layer(state).tanh()
        log_std = self.log_std_layer(state).tanh()
        # clamp the log standard deviation to be be in a balanced range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)  
        # get positive standard deviation
        std = torch.exp(log_std)

        # create normal distribution from mean and standard deviation 
        dist = torch.distributions.Normal(mean, std)
        # the sum of all discrete terms in samples of distribution
        z = dist.rsample()
        # activate z to get action in range of -1 and 1  
        action = z.tanh()
        
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return self.sample_normal(x)


class Critic(nn.Module):
    """Critic base Model."""

    def __init__(self, input_dim, seed=0, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc1_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class CriticQ(Critic):
    """Critic (Q-Function) Model."""

    def __init__(self, input_dim, seed=0, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticQ, self).__init__(input_dim, seed, fc1_units, fc2_units)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Value(Critic):
    """Value Model."""

    def __init__(self, input_dim, seed=0, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Value, self).__init__(input_dim, seed, fc1_units, fc2_units)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
