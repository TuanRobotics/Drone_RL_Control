# Drone_RL_Control
In this project, our idea is to use Reinforcement Learning algorithms to train a drone to fly through a gate whose position is known in advance, and then extend it into a full drone-racing problem. The RL algorithms we plan to use include PPO, SAC, and TD3

## Env:
[Examples](https://github.com/khlaifiabilel/reinforcement-learning-of-quadrotor-control/blob/master/gym_pybullet_drones/envs/multi_agent_rl/LeaderFollowerAviary.py)

## PPO: 
[PPO theorem](https://spinningup.openai.com/_/downloads/en/latest/pdf/?utm_source=chatgpt.com)
[PPO imlementation examples](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb#scrollTo=UT6VUBg-F8Zm)

## SAC
[SAC](https://janak-lal.com.np/solving-bipedal-walker-hardcore-challenge-with-soft-actor-critic-algorithm/)


## Achitectures: 
    - Lstm: [Paper](#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains)
    - Transformer
    - mlp model

## TODO List:
 - 2/12 - 3/12: Training Drone thrugate using lstm and mlpEncoder for SAC and TD3
 - 4/12 - 6/12: Training Drone racing using lstm and mlpEncoder for SAC and TD3