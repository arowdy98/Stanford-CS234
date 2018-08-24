# -*- coding: UTF-8 -*-

import os
import sys
import roboschool
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import gym.spaces
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import config
from OpenGL import GLU

# import logz


def build_mlp(
        mlp_input,
        output_size,
        scope,
        n_layers=config.n_layers,
        size=config.layer_size,
        output_activation=None):
    '''
    Build a feed forward network (multi-layer-perceptron, or mlp)
    with 'n_layers' hidden layers, each of size 'size' units.
    Use tf.nn.relu nonlinearity between layers.
    Args:
            mlp_input: the input to the multi-layer perceptron
            output_size: the output layer size
            scope: the scope of the neural network
            n_layers: the number of layers of the network
            size: the size of each layer:
            output_activation: the activation of output layer
    Returns:
            The tensor output of the network

    TODO: Implement this function. This will be similar to the linear
    model you implemented for Assignment 2.
    "tf.layers.dense" or "tf.contrib.layers.fully_connected" may be helpful.

    A network with n layers has n
      linear transform + nonlinearity
    operations before a final linear transform for the output layer
    (followed by the output activation, if it is not None).

    '''
    #######################################################
    #########   YOUR CODE HERE - 7-20 lines.   ############
    with tf.variable_scope(scope):
        x = mlp_input
        for layer in range(n_layers):
            x = tf.layers.dense(x, size, activation=tf.nn.relu)
        x = tf.layers.dense(x, output_size, activation=output_activation)
    return x
    #######################################################
    #########          END YOUR CODE.          ############


class PG(object):
    """
    Abstract Class for implementing a Policy Gradient Based Algorithm
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Policy Gradient Class

        Args:
                env: the open-ai environment
                config: class with hyperparameters
                logger: logger instance from logging module

        You do not need to implement anything in this function. However,
        you will need to use self.discrete, self.observation_dim,
        self.action_dim, and self.lr in other methods.

        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper-params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # discrete action space or continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.lr = self.config.learning_rate

        # build model
        self.build()

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph
        Set up the observation, action, and advantage placeholder

        TODO: add placeholders:
        self.observation_placeholder, type = tf.float32
        self.action_placeholder, type depends on the self.discrete
        self.advantage_placeholder, type = tf.float32

        HINT: In the case of a continuous action space, an action will be specified
        by self.action_dim float32 numbers.
        """
        #######################################################
        #########   YOUR CODE HERE - 8-12 lines.   ############
        obs_shape = self.env.observation_space.shape
        self.observation_placeholder = tf.placeholder(tf.float32,
                                                      shape=(None,) + obs_shape)
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int32,
                                                     shape=(None,))
        else:
            self.action_placeholder = tf.placeholder(tf.float32,
                                                 shape=(None, self.action_dim))

        # Define a placeholder for advantages
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None,))
        #######################################################
        #########          END YOUR CODE.          ############

    def build_policy_network_op(self, scope="policy_network"):
        """
        Build the policy network, construct the tensorflow operation to sample
        actions from the policy network outputs, and compute the log probabilities
        of the taken actions (for computing the loss later). These operations are
        stored in self.sampled_action and self.logprob. Must handle both settings
        of self.discrete.

        TODO:
        Discrete case:
            logits: the logits for each action
                HINT: use build_mlp
            self.sampled_action: sample from these logits
                HINT: use tf.multinomial + tf.squeeze
            self.logprob: compute the log probabilities of the taken actions
                HINT: 1. tf.nn.sparse_softmax_cross_entropy_with_logits computes
                         the *negative* log probabilities of labels, given logits.
                      2. taken actions are different than sampled actions!

        Continuous case:
            To build a policy in a continuous action space domain, we will have the
            model output the means of each action dimension, and then sample from
            a multivariate normal distribution with these means and trainable standard
            deviation.

            That is, the action a_t ~ N( mu(o_t), sigma)
            where mu(o_t) is the network that outputs the means for each action
            dimension, and sigma is a trainable variable for the standard deviations.
            N here is a multivariate gaussian distribution with the given parameters.

            action_means: the predicted means for each action dimension.
                HINT: use build_mlp
            log_std: a trainable variable for the log standard deviations.
            --> think about why we use log std as the trainable variable instead of std
            self.sampled_actions: sample from the gaussian distribution as described above
                HINT: use tf.random_normal
            self.lobprob: the log probabilities of the taken actions
                HINT: use tf.contrib.distributions.MultivariateNormalDiag

        """
        #######################################################
        #########   YOUR CODE HERE - 5-10 lines.   ############

        if self.discrete:
            action_logits = build_mlp(
                self.observation_placeholder,
                self.action_dim,
                scope,
                n_layers=self.config.n_layers,
                size=self.config.layer_size,
                output_activation=self.config.activation
            )
            self.sampled_action = tf.multinomial(action_logits, 1)
            self.sampled_action = tf.reshape(self.sampled_action, [-1,])
            self.logprob = - tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.action_placeholder,
                logits=action_logits)
        else:
            self.action_means = build_mlp(
                self.observation_placeholder,
                self.action_dim,
                scope,
                n_layers=self.config.n_layers,
                size=self.config.layer_size,
                output_activation=None)
            self.std = tf.get_variable(
                'std',
                shape=[1, self.action_dim],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=True
            )
            self.log_std = tf.log(tf.exp(self.std) + 1.)
            self.log_std = tf.tile(self.log_std,
                                   [tf.shape(self.action_means)[0], 1])
            self.sampled_action = tf.random_normal((1,),
                                                   self.action_means,
                                                   self.log_std)
            # self.sampled_action = tf.squeeze(self.sampled_action)
            # self.sampled_action = tf.clip_by_value(self.sampled_action,
            #                                        env.action_space.low[0],
            #                                        env.action_space.high[0]
            #                                        )
            self.dist = tf.contrib.distributions.MultivariateNormalDiag(
                self.action_means, self.log_std)
            self.logprob = self.dist.prob(self.action_placeholder)
            self.logprob = self.logprob
        #######################################################
        #########          END YOUR CODE.          ############

    def add_loss_op(self):
        """
        Sets the loss of a batch, the loss is a scalar

        TODO: Compute the loss for a given batch.
        Recall the update for REINFORCE with advantage:
        θ = θ + α ∇_θ log π_θ(s_t, a_t) A_t
        Think about how to express this update as minimizing a
        loss (so that tensorflow will do the gradient computations
        for you).

        You only have to reference fields of self that you have
        already set in previous methods.

        """

        ######################################################
        #########   YOUR CODE HERE - 1-2 lines.   ############
        self.loss = -self.logprob * self.advantage_placeholder
        self.loss = tf.reduce_mean(self.loss)
        #######################################################
        #########          END YOUR CODE.          ############

    def add_optimizer_op(self):
        """
        Sets the optimizer using AdamOptimizer
        TODO: Set self.train_op
        HINT: Use self.lr, and minimize self.loss
        """
        ######################################################
        #########   YOUR CODE HERE - 1-2 lines.   ############
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        #######################################################
        #########          END YOUR CODE.          ############

    def add_baseline_op(self, scope="baseline"):
        """
        Build the baseline network within the scope

        In this function we will build the baseline network.
        Use build_mlp with the same parameters as the policy network to
        get the baseline estimate. You also have to setup a target
        placeholder and an update operation so the baseline can be trained.

        Args:
                scope: the scope of the baseline network

        TODO: Set
        self.baseline,
            HINT: use build_mlp
        self.baseline_target_placeholder,
        self.update_baseline_op,
            HINT: first construct a loss. Use tf.losses.mean_squared_error.

        """
        ######################################################
        #########   YOUR CODE HERE - 4-8 lines.   ############
        self.baseline = build_mlp(
            self.observation_placeholder,
            1, 'baseline_net', config.n_layers, config.layer_size, None)
        self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None,))
        loss = tf.losses.mean_squared_error(self.baseline_target_placeholder,
                                            tf.squeeze(self.baseline))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.update_baseline_op = optimizer.minimize(loss)
        #######################################################
        #########          END YOUR CODE.          ############

    def build(self):
        """
        Build model by adding all necessary variables

        You don't have to change anything here - we are just calling
        all the operations you already defined to build the tensorflow graph.
        """

        # add placeholders
        self.add_placeholders_op()
        # create policy net
        self.build_policy_network_op()
        # add square loss
        self.add_loss_op()
        # add optimizer for the main networks
        self.add_optimizer_op()

        if self.config.use_baseline:
            self.add_baseline_op()

    def initialize(self):
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables

        You don't have to change or use anything here.
        """
        # create tf session
        self.sess = tf.Session()
        # tensorboard stuff
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_summary(self):
        """
        Tensorboard stuff.

        You don't have to change or use anything here.
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.

        You don't have to change or use anything here.
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        You don't have to change or use anything here.

        Args:
                rewards: deque
                scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        """
        Add summary to tfboard

        You don't have to change or use anything here.
        """

        fd = {
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

    def sample_path(self, env, num_episodes=None):
        """
        Sample path for the environment.

        Args:
                num_episodes:   the number of episodes to be sampled
                  if none, sample one batch (size indicated by config file)
        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.config.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = 0.
                reward = 0.
                done = False
                action = self.sess.run(
                    self.sampled_action,
                    feed_dict={self.observation_placeholder: states[-1][None]})[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if (done or step == self.config.max_ep_len - 1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {"observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
          paths: recorded sampled path.  See sample_path() for details.

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        TODO: compute and return G_t for each timestep. Use config.gamma.
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            #######################################################
            #########   YOUR CODE HERE - 5-10 lines.   ############
            path_returns = []
            episode_len = len(rewards)
            for time_step in range(episode_len):
                tail_length = episode_len - time_step
                gammas = np.logspace(0, tail_length, num=tail_length,
                                     base=self.config.gamma, endpoint=False)
                return_t = np.multiply(rewards[time_step:],gammas)
                return_t = return_t.sum()
                path_returns.append(return_t)
            #######################################################
            #########          END YOUR CODE.          ############
            all_returns.append(path_returns)
        returns = np.concatenate(all_returns)

        return returns

    def calculate_advantage(self, returns, observations):
        """
        Calculate the advantage
        Args:
                returns: all discounted future returns for each step
                observations: observations
                  Calculate the advantages, using baseline adjustment if necessary,
                  and normalizing the advantages if necessary.
                  If neither of these options are True, just return returns.

        TODO:
        If config.use_baseline = False and config.normalize_advantage = False,
        then the "advantage" is just going to be the returns (and not actually
        an advantage).

        if config.use_baseline, then we need to evaluate the baseline and subtract
          it from the returns to get the advantage.
          HINT: 1. evaluate the self.baseline with self.sess.run(...

        if config.normalize_advantage:
          after doing the above, normalize the advantages so that they have a mean of 0
          and standard deviation of 1.

        """
        adv = returns
        #######################################################
        #########   YOUR CODE HERE - 5-10 lines.   ############
        if self.config.use_baseline:
            baselines = self.sess.run(
                self.baseline,
                feed_dict={
                    self.observation_placeholder: observations
                }
            )
            adv -= baselines.squeeze()
        if self.config.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-12)

        #######################################################
        #########          END YOUR CODE.          ############
        return adv

    def update_baseline(self, returns, observations):
        """
        Update the baseline

        TODO:
          apply the baseline update op with the observations and the returns.
        """
        #######################################################
        #########   YOUR CODE HERE - 1-5 lines.   ############
        self.sess.run(
            self.update_baseline_op,
            feed_dict={
                self.observation_placeholder: observations,
                self.baseline_target_placeholder: returns
            }
        )
        #######################################################
        #########          END YOUR CODE.          ############

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_eval = 0
        last_record = 0
        scores_eval = []

        self.init_averages()
        scores_eval = []  # list of scores computed at iteration time

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            scores_eval = scores_eval + total_rewards
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.update_baseline(returns, observations)
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.observation_placeholder: observations,
                self.action_placeholder: actions,
                self.advantage_placeholder: advantages})

            # tf stuff
            if (t % self.config.summary_freq == 0):
                self.update_averages(total_rewards, scores_eval)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)
            #if self.config.record and (last_record > self.config.record_freq):
            if self.config.record and (t % self.config.record_freq == 0 ):
                self.logger.info("Recording...")
                #last_record = 0
                self.record()

        self.logger.info("- Training done.")
        export_plot(scores_eval, "Score", self.config.env_name, self.config.plot_output)

    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None: env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # initialize
        self.initialize()
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()


if __name__ == '__main__':
    env = gym.make(config.env_name)
    # train model
    model = PG(env, config)
    model.run()
