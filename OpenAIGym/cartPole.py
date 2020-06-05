import gym

env = gym.make('CartPole-v0')
print('INITIAL OBSERVATION', end=": ")
observation = env.reset()
print(observation)

for _ in range(10000):
	env.render()
	# take a random action
	cart_pos, cart_vel, pole_ang, ang_vel = observation
	if pole_ang > 0:
		action = 1
	else:
		action = 0
	observation, reward, done, info = env.step(action)
	print("observation:" ,end=" ")
	print(observation)
	print("reward:" ,end=" ")
	print(reward)
	print("done:" ,end=" ")
	print(done)
	print("info:" ,end=" ")
	print(info)
	# returns inputs, reward, done (if it needs to reset), info (dict)