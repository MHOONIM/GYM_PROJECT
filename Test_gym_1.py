import numpy as np
import tensorflow as tf

import gym


env = gym.make('MountainCar-v0')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
t = 0
# print(env.reset())
# print(env.action_space.sample())

for i in range(1):
    observation = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminate, truncrated, info = env.step(action)
    print(observation[0].reshape((1, 2)).shape)
    print(next_observation)
    print(action)
    print(reward)
    print(terminate)
    print(truncrated)
    print(info)
    print(env.action_space.n)

# while True:
#     t += 1
#     observation = env.reset()
#     env.render()
#     # print(observation)
#     action = env.action_space.sample()
#     print(env.action_space)
#     # if done:
#     #     print("Episode finished after {} timesteps".format(t+1))
#     # break
#     env.close()
