### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import gym.spaces
import time
from lake_envs import *
import random 

np.set_printoptions(precision=3)

def isover(V,V_new,tol):
	if np.all(np.abs(V - V_new) < tol) :    #np.sum(np.sqrt(np.square(V_new-V))) < tol
		return 1
	return 0

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	V=np.zeros(nS)
	V_new=V.copy()
	i=0
	for i in range(max_iteration):
		V=V_new.copy()
		V_new = np.zeros(nS, dtype=float)
		for state in range(nS):
			for probability, nextstate, reward, terminal in P[state][policy[state]]:
				V_new[state] += probability * (reward + gamma * V[nextstate])
		if isover(V,V_new,tol) :
			break
	return V_new

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	P_new = np.zeros(nS, dtype=int)
	for state in range(nS):
		B=np.zeros(nA,dtype=float)
		q=-99
		for action in range(nA):
			for probability, nextstate, reward, terminal in P[state][action]:
				B[action] += probability * (reward + gamma * value_from_policy[nextstate])
			if(B[action]>q):
				q=B[action]
				P_new[state]=action
			elif(q == B[action]):
				if random.random() < 0.5:
					P_new[state]=action
	return P_new

def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS,dtype=float)
	policy = np.zeros(nS, dtype=int)
	for s in range(nS):
		policy[s]=s%nA
	for i in range(max_iteration):
		V_new=policy_evaluation(P, nS, nA, policy, gamma)
		policy_new=policy_improvement(P, nS, nA, V_new, policy, gamma)
		if isover(V,V_new,tol) :
			break
		V=V_new.copy()
		policy=policy_new.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	return V, policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS,dtype=float)
	policy = np.zeros(nS, dtype=int)
	for i in range(max_iteration):
		V_next=np.zeros(nS,dtype=float)
		for s in range(nS):
			for a in range(nA):
				q=0
				for probability, nextstate, reward, terminal in P[s][a]:
					q += probability * (reward + gamma * V[nextstate])
				if V_next[s] < q:
					V_next[s] = q
		if isover(V,V_next,tol):
			break
		V = V_next.copy()
	policy=policy_improvement(P, nS, nA, V_next, policy, gamma)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	return V_next, policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	#print env.__doc__
	#print "Here is an example of state, action, reward, and next state"
	#example(env)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=600, tol=1e-4)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=600, tol=1e-4)
	print('Policy Iteration')
	print('  Optimal Value Function: %r' % V_pi)
	print('  Optimal Policy:         %r' % p_pi)
	print('Value Iteration')
	print('  Optimal Value Function: %r' % V_vi)
	print('  Optimal Policy:         %r' % p_vi)
	render_single(env,p_pi)
	render_single(env,p_vi)
	print('\n##########\n##########\n\n')