from unityagents import UnityEnvironment
import numpy as np
import numpy as np
from multi_sac_agent import Agent
from collections import deque
import torch
from pprint import PrettyPrinter

def init_environment(hyperparameters=""):
    # initialise the headless unity environment
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

    # get the default environment, called brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # retrieve the facts about the unity environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    # initialize agent with environment facts and hyperparameters for actor critic model components
    if hyperparameters:
        agent = Agent(state_size=state_size, action_size=action_size, random_seed=0, num_agents=num_agents, hyperparameters=hyperparameters)
    else:
        agent = Agent(state_size=state_size, action_size=action_size, random_seed=0, num_agents=num_agents)
    init_output = \
    """##Reacher Environment##

# Environment Details
- Number of Agents: {num_agents}
- Size of Action (Continuous): {action_size} 
- Number of state variables: {state_size}

# Hyper Parameters
{hyperparameters}
    """.format(
            num_agents=num_agents, 
            action_size=action_size,
            state_size=state_size, 
            hyperparameters=PrettyPrinter().pformat(agent.hyperparameters)
    )

    print(init_output)
    
    return agent, env, brain_name


def multi_sac_runtime(n_episodes=10000, reward_goal=0.5, max_t=10000, window_size=100):
    """Runtime for Multi-Agent Soft Actor-Critic. 
    Runs for maximum of n_episodes the ddpg agent against a Unity Reacher Environment.
    Either the number of n_episodes is reached or the mean rewards in reward_goal to finish the runtime.
    
    Params
    ======
        n_episodes (int): Maximum number of training episodes
        reward_goal (int): Mean reward agent has to reach
        max_t (int): Maximum number of timesteps per episode
        window_size (int): Number of past episodes used for mean rewards"""
   
    agent, env, brain_name = init_environment()
    scores = []
    scores_deque = deque(maxlen=window_size)
    num_agents = 2
    step = 0
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = [0]*num_agents
        
        for t in range(max_t):
            step += 1
            # select an action
            action = agent.act(state, step)
            # run action in used Unity Environment
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            # run action in Multi SAC agent (also learning models)
            agent.step(state, action, reward, next_state, done, t)
            score = [(score[i] + reward[i]) for i in range(num_agents)]
            state = next_state
            if np.any(done):
                break
        
        # save score for total runtime
        scores.append(score)
        # save score to mean reward calculation over n episodes
        scores_deque.append(max(score))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        if i_episode % 5 == 0:
            with open('last_scores.txt', 'w') as score_file:
                for element in scores:
                    score_file.write(str(element))
                    score_file.write("\n")

        if np.mean(scores_deque)>=reward_goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_01.state_dict(), 'checkpoint_critic_01.pth')
            torch.save(agent.critic_02.state_dict(), 'checkpoint_critic_02.pth')
            torch.save(agent.value_local.state_dict(), 'checkpoint_value_local.pth')
            with open('final_scores.txt', 'w') as score_file:
                for element in scores:
                    score_file.write(str(element))
                    score_file.write("\n")
            break
            
    return scores


if __name__ == "__main__":
    scores = multi_sac_runtime(n_episodes=10000)
