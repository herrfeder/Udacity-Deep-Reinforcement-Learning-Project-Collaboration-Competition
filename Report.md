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
 'policy_update': 2,                                                                                                                                                                                               
 'tau': 0.005}
```

  * The training was done with a `Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz` in a headless VPS Machine.
  * It took about 2 hours and 887 episodes to finish the training successfully:
  
![](images/screenshot_finished_training.png)

![](images/sac_training_diagram.png)

# Used Code

I've used the [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) example as the skeleton for my implementation and tried to convert it for using a Multi-Agent-SAC instead of a DDPG for a single agent. I used the following sources to collect ideas and implement my version, that was able to reach the desired Reward:

  * https://github.com/MrSyee/pg-is-all-you-need
  * https://spinningup.openai.com/en/latest/algorithms/sac.html
  * https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665

## SAC Algorithm

The Soft Actor-Critic (Algorithm) is a Off-Policy-Algorithm which leverages Q-Learning to learn an optimal policy. 
In contrast to DDPG, the SAC algorithm tries to maximize the reward while also maximizing the entropy of the resulting policy to have the optimal tradeoff between exploration and learning stability. To measure the optimal amount of entropy, the Kullback-Lieber Divergence is used. (Great Article about KL-Divergence in SAC -> https://towardsdatascience.com/entropy-in-soft-actor-critic-part-2-59821bdd5671)
The most important parameter to control this tradeoff is the temperature parameter Alpha, which can also be learned during environment training and treated as a dynamic value. (As stated here: https://arxiv.org/pdf/1812.05905.pdf). To keep things simple I treat this Alpha as a static hyperparameter.

The actor can be seen as the Policy Network, the critic as the Network to approximate the Q-Function ... value
In this implemention two parallel Critics for the Q-Network are used to handle the problems of overestimation. 

"""
Both are used together to calculate the next-state Q-Values and the critic will try minimize the loss between the updated and the original Q value. With this information the actor will try to maxize the expected return by optimizing the policy.
Both elements have additional target networks, which can be seen as time-delayed representations of the actor/critic local network. Isolate the local and the target and soft updating the target networks helps in improving learning performance and preventing divergence.
"""

For applying actual deep learning, we need some kind of Buffer, from with training data can be sampled. This is done using a Replay Buffer. One "row" of data is a tuple of `(state, next_state, action, reward, done)`. 

The training and learning is happening in steps over the episodes happening in the environment.
  * Initially, when starting:
    * the actor, value and critic networks are initialised with random weights
    * the Replay Buffer gets initialised
  * For the beginning of every episode:
    * 
  * During each step of a episode:
    * according to the current policy an action is executed, the reward and and the next state gets collected
    * each step we are storing actions and states into the Replay Buffer and take a sample from it
    * this sample is used to update the critic by minimizing the loss between original and updated Q-Values
    * and updating the actor by applying Gradient Ascent to the sampled policy gradient
    * in the end of the step the target networks gets updated


## Model

### Modifications

The used model architecture corresponds with the pendulum example and an additional layer for Batch Normalization after the ReLu-Activation of the first Linear Layer. I tried several locations for the Batch Normalization layer. Especially the topic of whether placing the Batch Normalization before or after an ReLu activation is subject of many discussions like this one: https://forums.fast.ai/t/why-perform-batch-norm-before-relu-and-not-after/81293/4

After checking both alternatives, the Batch Normalization before ReLu-Activation shows much better results.
In combination with the BATCH_SIZE of 256 the learning had it's best performance with a lenght of 128 for both fully connected linear layers in the Actor and Critic Model.


## DDPG Agent and Replay Buffer

### Modifications

The used implementation of the DDPG agent corresponds also strongly with the pendulum example. I changed one important thing, I removed the `OUNoise` completely and replaced it with a static value for the noise function. I found the great article https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3 from another student of this course. He experimented with different sources and distributions of noise and came to the conclusion that in the case of the Reacher environment a static noise value works nearly as good as the Ornstein-Uhlenbeck Noise generator.
To reduce the number of influencing variables, I tried to train my DDPG using a static scalar, like Soeren did and got a faster Training Process, than with the `OUNoise before`.

# Future Work

## Influence of the Noise Process

  * Like the Student Soeren discovered, the Noise process has a significant impact on the learning performance. It seems, that static values, time-dependent or distance-dependent noise processes are feasible alternatives to optimize the exploring of the agent. I'm pretty sure, this can be optimized further and can increase the learning performance and the agent stability.
  
## Multi-Agent Training

 * Using multiple agents can dramatically decrease the necessary training time. How the agents are orchestrated in the episodes, how their values gets collected and the networks updated can play a huge role for training performance.
 
## Prioritized Replay

  * Right now our Replay Buffer doesn't any selection of very good training episodes. Like seen in this course before, Prioritized Replay could be applied to increase the training performance.

