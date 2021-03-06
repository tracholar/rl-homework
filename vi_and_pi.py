#coding:utf-8
### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters


import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


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
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	epsilon = np.inf
	for it in range(max_iteration):
		if epsilon < tol:
			break
		max_dv = 0
		for s in range(nS):  # for each state s
			max_value = - np.inf
			max_action = 0
			for a in range(nA): # for each action
				action_value = 0
				for probability, nextstate, reward, terminal in P[s][a]:
					action_value += probability * (reward + gamma * V[nextstate])

				if max_value < action_value:
					max_value = action_value
					max_action = a
			if abs(V[s] - max_value) > max_dv:
				max_dv = abs(V[s] - max_value)
			V[s] = max_value
			policy[s] = max_action
		if epsilon > max_dv:
			epsilon = max_dv
		#print '\rValue Iteration %d, epsilon=%f' % (it, epsilon),

	return V, policy


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
	V = np.zeros(nS)
	epsilon = np.inf
	for i in range(max_iteration):
		if epsilon < tol:
			break
		max_dv = 0
		for s in range(nS):
			a = policy[s]
			probability, nextstate, reward, terminal = P[s][a][0]
			state_value = reward
			for probability, nextstate, reward, terminal in P[s][a]:
				state_value += gamma * probability * V[nextstate]
			if abs(V[s] - state_value) > max_dv:
				max_dv = abs(V[s] - state_value)
			V[s] = state_value
		if epsilon > max_dv:
			epsilon = max_dv
		print '\rpolicy_evaluation iteration %d, epsilon = %f' % (i, epsilon),
	print 'Norm of V = ', np.linalg.norm(V)

	return V


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
	Q = np.zeros(shape=(nS, nA))
	new_policy = np.zeros(nS, dtype='int')

	for s in range(nS):
		max_q = 0
		max_a = 0
		for a in range(nA):
			Q[s][a] = 0.0
			for probability, nextstate, reward, terminal in P[s][a]:
				Q[s][a] += probability * (reward + gamma *  value_from_policy[nextstate])

			if Q[s][a] > max_q:
				max_q = Q[s][a]
				max_a = a

		new_policy[s] = max_a

	return new_policy


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
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	for it in range(max_iteration):
		V[:] = policy_evaluation(P, nS, nA, policy, gamma)
		new_policy = policy_improvement(P, nS, nA, V, policy, gamma)
		if np.max(np.abs(policy - new_policy)) == 0:
			break
		print 'policy_iteration %d, diff policy = %d' % (it, np.max(np.abs(policy - new_policy)) )
		policy[:] = new_policy
	return V, policy



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
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print env.__doc__
	print "Here is an example of state, action, reward, and next state"
	example(env)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	print ''
	for i in range(4):
		for j in range(4):
			print V_vi[i*4+j],
		print ''

	act = [u'⬅️',u'⬇️',u'➡️',u'⬆️']
	for i in range(4):
		for j in range(4):
			print act[p_vi[i*4+j]],
		print ''

	render_single(env, p_vi)
	#print env.P

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	render_single(env, p_pi)
