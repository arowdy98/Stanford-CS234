import numpy as np
import gym
import time
from lake_envs import *
import matplotlib.pyplot as plt
import sys

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99):
	# num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
	"""Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
	Update Q at the end of every episode.

	Parameters
	----------
	env: gym.core.Environment
	Environment to compute Q function for. Must have nS, nA, and P as
	attributes.
	num_episodes: int 
	Number of episodes of training.
	gamma: float
	Discount factor. Number in range [0, 1)
	learning_rate: float
	Learning rate. Number in range [0, 1)
	e: float
	Epsilon value used in the epsilon-greedy method. 
	decay_rate: float
	Rate at which epsilon falls. Number in range [0, 1)

	Returns
	-------
	np.array
	An array of shape [env.nS x env.nA] representing state, action values
	"""

	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	Q=np.zeros((env.nS, env.nA))
	avg_rewards=np.zeros((num_episodes,))
	tot_reward=0
	for i in range(num_episodes):
		s=env.reset()
		for j in range(400):
			if np.random.rand()>e:
				a=np.argmax(Q[s])
			else:
				a=np.random.randint(env.nA)
			nexts, reward, done, info = env.step(a)
			Q[s][a]+=lr*(reward+gamma*np.max(Q[nexts]) - Q[s][a])
			tot_reward+=reward
			s=nexts
			if done:
				break
		avg_rewards[i]=tot_reward/(i+1)
		print "Total reward until episode",i+1,":",tot_reward
		sys.stdout.flush()
		if i%10 == 0:
			e=e*decay_rate
	return Q,avg_rewards

def render_single_Q(env, Q):
	"""Renders Q function once on environment. Watch your agent play!

	Parameters
	----------
	env: gym.core.Environment
	  Environment to play Q function on. Must have nS, nA, and P as
	  attributes.
	Q: np.array of shape [env.nS x env.nA]
	  state-action values.
	"""

	episode_reward = 0
	state = env.reset()
	done = False
	while not done:
		env.render()
		time.sleep(0.2) # Seconds between frames. Modify as you wish.
		action = np.argmax(Q[state])
		state, reward, done, _ = env.step(action)
		episode_reward += reward

	print "Episode reward: %f" % episode_reward

def evaluate_q(env, Q, num_episodes=100):
	tot_reward=0
	for i in range(num_episodes):
		episode_reward = 0
		state = env.reset()
		done = False
		while not done:
			action = np.argmax(Q[state])
			state, reward, done, _ = env.step(action)
			episode_reward += reward
		tot_reward+=episode_reward
	print "Total",tot_reward,"reward in",num_episodes,"episodes"
	print "Average Reward:",tot_reward/num_episodes

# Feel free to run your own debug code in main!
def main():
	num_episodes=3000
	env = gym.make('Stochastic-4x4-FrozenLake-v0')
	Q, avg_rewards = learn_Q_QLearning(env,num_episodes)
	render_single_Q(env, Q)
	evaluate_q(env, Q, 200)
	evaluate_q(env, Q, 200)
	evaluate_q(env, Q, 200)
	plt.plot(range(num_episodes),avg_rewards)
	plt.show()

if __name__ == '__main__':
    main()
