import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')
# number of observations (16 [4 * 4])
STATES = env.observation_space.n
# all possible actions (4)
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

# number of runs
EPISODES = 6000
# max steps per episode
MAX_STEPS = 100

# eta
LEARNING_RATE = 0.81

GAMMA = 0.96

# chance of taking a random action
epsilon = 0.9

# if np.random.uniform(0, 1) < epsilon:
# 	action = env.action_space.sample()
# else:
# 	action = np.argmax(Q[state, :])

# update Q values
# Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])

RENDER = False

rewards = []

for episode in range(EPISODES):
	print(episode)
	state = env.reset()
	for _ in range(MAX_STEPS):
		if RENDER:
			env.render()
		if np.random.uniform(0, 1) < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state, :])
		new_state, reward, done, _ = env.step(action)
		print()
		print(state)
		Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])
		state = new_state
		if done:
			rewards.append(reward)
			epsilon -= 0.001
			break
print(Q)
print(f"Average reward: {np.mean(rewards)}")

import matplotlib.pyplot as plt

avg_rewards = []
for i in range(0, len(rewards), 100):
	avg_rewards.append(np.mean(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel("average reward")
plt.xlabel("episodes (100\'s)")
plt.show()