import gymnasium as gym 
import random
import gym_examples
import time

# Action Sets
rotation_0 = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
rotation_1 = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37]
rotation_2 = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
rotation_3 = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

total_reward = 0
env = gym.make("gym_examples/Tetris-v0", width = 10, height = 20)
observation, info = env.reset()
print("reset", observation, len(observation))
actions = rotation_3 #[1, 9, 17,25,33]
action_index=0
for _ in range(1000):
   env.render()
   print("Tetrimino", _)
   action = actions[action_index] #env.action_space.sample() 
   print("action", action)
   observation, reward, terminated, truncated, info = env.step(action)
   print("obs",  observation)
   total_reward += reward
   time.sleep(2)

   if action_index==len(actions)-1:
      action_index=0
      #break
   else:
      action_index+=1

   if terminated or truncated:
      observation, info = env.reset()
      print("---------------------------------------------------------------------------")
      print("Ended", total_reward)
      print("---------------------------------------------------------------------------")
      total_reward = 0
      #break
env.close()