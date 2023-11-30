# Training Specs

I used the Tennis Unity Environment which is a Multi-Agent Environment with **two competing agents**. As the training process took several hours it was much more handy for me to run the training in a headless environment via a python script instead a jupyter notebook.
The hyperparameters where choosen with some assumptions but in the end the best results where found by trying different combinations. There are some tricks (like initial random action steps) , for example mentioned by OpenAI in their SAC documentation, which obviously will speed up the training process. I tried to stay as basic as possible and tried to avoid as many "tricks" while having feasible training results.

```
##Tennis Environment##

# Environment Details                                                                                                                                                                                              
- Number of Agents: 2                                                                                                                                                                                              
- Size of Action (Continuous): 2                                                                                                                                                                                   
- Number of state variables: 24                                                                                                                                                                                    
                                                                                                                                                                                                                   
# Hyper Parameters                                                                                                                                                                                                 
{'batch_size': 64,                                                                                                                                                                                                 
 'buffer_size': 10000,                                                                                                                                                                                             
 'entropy_weight': 0.00025,                                                                                                                                                                                        
 'gamma': 0.99,                                                                                                                                                                                                    
 'initial_rand_steps': 100,                                                                                                                                                                                        
 'learning_rate': 0.0003,                                                                                                                                                                                          
 'lin_full_con_01': 256,                                                                                                                                                                                           
 'lin_full_con_02': 256,                                                                                                                                                                                           
 'tau': 0.005}
```

  * The training was done with a `Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz` in a headless VPS Machine.
  * It took about 2 hours and 1003 episodes to finish the training successfully:
  
![](images/screenshot_finished_training.png)

![](images/sac_training_diagram.png)

# Used Code

I've used the [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) example as the skeleton for my implementation and tried to convert it for using a Multi-Agent-SAC instead of a DDPG for a single agent. I used the following sources to collect ideas and implement my version, that was able to reach the desired Reward:

  * https://github.com/MrSyee/pg-is-all-you-need
    * Implemention of the SAC algorithm
  * https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
    * Implementation of the SAC algorithm
  * https://github.com/kantologist/multiagent-sac/blob/master/agents/masac_agent.py
    * How to deal with multiple agents and store their experiences

# SAC Algorithm (in easy words as far as my understanding reaches :) )

  * The Soft Actor-Critic (Algorithm) is a Off-Policy-Algorithm which leverages Q-Learning to learn an optimal policy. 
  * In contrast to DDPG, the SAC algorithm tries to maximize the reward while also maximizing the entropy of the resulting policy to have the optimal tradeoff between exploration and learning stability. To measure the optimal amount of entropy, the Kullback-Lieber Divergence is used. (Great Article about KL-Divergence in SAC -> https://towardsdatascience.com/entropy-in-soft-actor-critic-part-2-59821bdd5671)
  * The most important parameter to control this tradeoff is the temperature parameter Alpha, which can also be learned during environment training and treated as a dynamic value. (As stated here: https://arxiv.org/pdf/1812.05905.pdf). To keep things simple I treat this Alpha as a static hyperparameter.
  * The actor can be seen as the Policy Network, the critic as the Network to approximate the Q-Function.
  * In this implemention two parallel Critics for the Q-Network are used to handle the problems of overestimation.
  * The original SAC implementation also leverages an additional Value-Network (which is also used in my implementation), which improved the stability of learning (like described in this issue https://github.com/hill-a/stable-baselines/issues/270). In the meantime the additional Value network was removed in the more recent reference implementations: https://github.com/rail-berkeley/softlearning/
    * this Value network is the only network that utilizes a target network

  * For applying actual deep learning, we need some kind of Buffer, from with training data can be sampled. This is done using a Replay Buffer. One "row" of data is a tuple of `(state, next_state, action, reward, done)`. 

## Training Process

The training and learning is happening in steps over the episodes happening in the environment.
  * Initially, when starting:
    * the actor, value and critic networks are initialised with random weights
    * the Replay Buffer gets initialised
  * During each step of a episode:
    * the agent selects an action which is executed against the environment doing an environment step
      * for the first N steps, the action is sampled from a uniform random distribution as it improves learning performance very strongly (according to OpenAI)
    * the reward and state (current state and next state) gets collected in the Replay Buffer for each agent
    * when the Replay Buffer reaches the size of given Batch Size the Updating/Learning Process is started and will be executed for every consecutive step:
      * the action and log probability of this action is sampled again from the agent
      * both Critics (Q-Networks) are updated using Gradient Descent and gets influenced by the Prediction of the target Value Network
      * the best prediction from both critics is collected
      * this Q-Prediction and the collected log probability is used to set the target for the Value network, which will be updated using Gradient Descent
      * the actor loss (Policy Net) is now calculated using the Predictions for the Q-Network, Value network and the Log Probability of the sampled action
      * the actor loss gets minimized, means the reward gets maximized, by utilizing Gradient Ascent
      * in the end of the step the Value target networks gets updated from the updated Value Network


# Future Work

## Automatic Learning of the temperature parameter Alpha

  * Instead of using a static value for the temperature parameter Alpha, it can be learned with own neural network layers, which will be incluenced with the log probability that comes from the new sampled action in the policy network
  * This should improve the learning performance and will reduce the number of hyperparameters that needs to be adjusted for other environments

## Emphasized/Prioritized Replay

  * Like with DDPG before, we are using right now a simple Replay Buffer without any adaptive Sampling. For example this paper (https://arxiv.org/pdf/1906.04009.pdf) is implementing a mechanism that will reduce the sampling range over time and will more and more emphasize the most recent experience
  * this sounds like a good way to improve the learning performance

## State-Of-The-Art DRL vs. Evolutionary Algorithms

  * as far as I know is Soft Actor-Critic still recognized as the state-of-the-art approach for off-policy Training in Continuous Spaces
  * therefore it's maybe promising to look at completely different ways of achieving good results in competetive learning model environments like Evolutionalary Algorithms
  * maybe it's a good way to even combine Evolution-based Algorithms with Soft Actor-Critic

