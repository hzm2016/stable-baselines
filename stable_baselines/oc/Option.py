#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of deterministic option, including deterministic policy
and termination function, which are parameterized by different parameters.

"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)

# the maximum times to double the weights
DOUBLE_TIME = 10
LAYER1 = 64
LAYER2 = 64

# Avoid NaN (prevents division by zero or log of zero)
EPS = 1e-6
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(input_, mu_, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def mlp(input_ph, layers, activ_fn=tf.nn.relu, layer_norm=False):
    """
    Create a multi-layer fully connected neural network.

    :param input_ph: (tf.placeholder)
    :param layers: ([int]) Network architecture
    :param activ_fn: (tf.function) Activation function
    :param layer_norm: (bool) Whether to apply layer normalization or not
    :return: (tf.Tensor)
    """
    output = input_ph
    for i, layer_size in enumerate(layers):
        output = tf.layers.dense(output, layer_size, name='fc' + str(i))
        if layer_norm:
            output = tf.keras.layers.LayerNormalization(output, center=True, scale=True)
            # output = tf.contrib.layers.layer_norm(output, center=True, scale=True)
        output = activ_fn(output)
    return output


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the ouput of the gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)

    return deterministic_policy, policy, logp_pi


class Option:

    def __init__(self, session, state_dim, action_dim, ordinal, tau=1e-2, learning_rate=1e-3):
        """
        :param learning_rate: (learning_rate_policy, learning_rate_termin)
        :param ordinal: the name to tell different options apart
        """
        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.ad = action_dim
        self.ord = ordinal
        lrp, lrt = learning_rate

        # some placeholders
        self.s = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')
        self.qg = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='q_gradient')
        self.adv = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='advantage')

        # evaluation and target scope
        ep_scope = 'eval_policy_' + str(ordinal)
        tp_scope = 'target_policy_' + str(ordinal)
        te_scope = 'termination_' + str(ordinal)

        # evaluation and target network
        self.a = self._option_net(scope=ep_scope, trainable=True)
        self.a_ = self._option_net(scope=tp_scope, trainable=False)
        self.p = self._termination_net(scope=te_scope, trainable=True)

        # soft update
        ep_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ep_scope)
        tp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tp_scope)
        te_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=te_scope)
        self.update = [tf.assign(t, e) for t, e in zip(tp_params, ep_params)]

        # define optimizer
        pg = tf.gradients(ys=self.a, xs=ep_params, grad_ys=self.qg)
        tg = tf.gradients(ys=self.p, xs=te_params, grad_ys=-self.adv)
        self.pop = tf.train.AdamOptimizer(-lrp).apply_gradients(zip(pg, ep_params))
        self.top = tf.train.AdamOptimizer(-lrt).apply_gradients(zip(tg, te_params))

        self.prob = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.diff = tf.reduce_max(tf.abs(self.prob - self.p))
        self.loss = tf.nn.l2_loss(self.prob - self.p)
        self.oop = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

        self.train_counter = 0

    def train(self, state_batch, q_gradient_batch, advantage_batch=None):
        """Train the policy and termination function"""

        self.sess.run(self.pop, feed_dict={
            self.s: state_batch,
            self.qg: q_gradient_batch
        })

        if advantage_batch is not None:
            self.sess.run(self.top, feed_dict={
                self.s: state_batch,
                self.adv: advantage_batch
            })

        self.train_counter += 1

        if self.train_counter == 10:
            self.sess.run(self.update)
            self.train_counter = 0

    def choose_action(self, state):
        """Choose action"""

        return self.sess.run(self.a, feed_dict={
            self.s: state[np.newaxis, :]
        })[0]

    def get_prob(self, state):
        """Get termination probability of current state"""

        return self.sess.run(self.p, feed_dict={
            self.s: state[np.newaxis, :]
        })[0]

    def get_actions(self, state_batch):
        """Get target actions"""

        return self.sess.run(self.a, feed_dict={
            self.s: state_batch
        })

    def get_target_actions(self, state_batch):
        """Get target actions"""

        return self.sess.run(self.a_, feed_dict={
            self.s: state_batch
        })

    def _option_net(self, scope, trainable):
        """Generate evaluation/target option network"""

        # tf.nn.relu
        activation = tf.nn.tanh

        with tf.variable_scope(scope):

            """the first layer"""
            x = tf.layers.dense(self.s, LAYER1, activation=activation,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense1')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch1')

            """the second layer"""
            x = tf.layers.dense(x, LAYER2, activation=activation,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense2')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch2')

            """last layer"""
            action = tf.layers.dense(x, self.ad, activation=tf.nn.tanh,
                                     kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                     trainable=trainable, name='action')

        return action

    def _termination_net(self, scope, trainable):
        """Generate evaluation/target option network"""

        with tf.variable_scope(scope):

            x = tf.layers.dense(self.s, 32, activation=tf.nn.relu,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense1')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch1')
            prob = tf.layers.dense(x, 1, activation=tf.nn.sigmoid,
                                   kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                   trainable=trainable, name='prob')

        return prob

    def render(self):
        """Render option and termination function"""

        fig, (ax0, ax1) = plt.subplots(1, 2)

        num = 100
        delta = 2.0 / num
        sta = -np.ones((num * num, 2)) + delta * 0.5
        u = np.zeros((num, num))
        v = np.zeros((num, num))
        p = np.zeros((num, num))

        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([j * delta, i * delta])
                sta[o] += s

        a = self.sess.run(self.a, feed_dict={
            self.s: sta
        })
        p1 = self.sess.run(self.p, feed_dict={
            self.s: sta
        })

        for i in range(num):
            for j in range(num):
                o = i * num + j
                u[i, j] = a[o, 0]
                v[i, j] = a[o, 1]
                p[i, j] = p1[o]

        V = (u * u + v * v) ** 0.5
        x = np.linspace(-1.0, 1.0, num + 1)
        ax0.streamplot(sta[:num, 0], sta[:num, 0], u, v, color=1.4-V)
        im0 = ax0.pcolor(x, x, V, cmap='jet')
        ax0.set_title('intra-policy')
        fig.colorbar(im0, ax=ax0)
        im1 = ax1.pcolor(x, x, p, cmap='jet')
        ax1.set_title('termination function')
        fig.colorbar(im1, ax=ax1)

        fig.tight_layout()
        plt.show()

    def pretrain(self):
        """Pretrain termination function(need to be designed specifically)"""

        ord = self.ord
        num = 50
        delta = 2.0 / (num - 1)
        test_state = -np.ones((num * num, 2))
        test_label = np.ones((num * num, 1))
        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([i * delta, j * delta])
                test_state[o] += s
                if ord == 0:
                    if test_state[o, 0] > 0 and test_state[o, 1] > 0:
                        test_label[o] = 0.0
                elif ord == 1:
                    if test_state[o, 0] < 0 < test_state[o, 1]:
                        test_label[o] = 0.0
                elif ord == 2:
                    if test_state[o, 0] < 0 and test_state[o, 1] < 0:
                        test_label[o] = 0.0
                elif ord == 3:
                    if test_state[o, 1] < 0 < test_state[o, 0]:
                        test_label[o] = 0.0

        bound = 1e-2
        while True:
            self.sess.run(self.oop, feed_dict={
                self.s: test_state,
                self.prob: test_label
            })
            a = self.sess.run(self.diff, feed_dict={
                self.s: test_state,
                self.prob: test_label
            })
            if a < bound:
                break


class Soft_Option(object):
    def __init__(self, session, state_dim, action_dim, ordinal, tau=1e-2,
                 learning_rate=1e-3, policy_kwargs=None, layers=None, layer_norm=False):

        """
        :param learning_rate: (learning_rate_policy, learning_rate_termin)
        :param ordinal: the name to tell different options apart
        """

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.ad = action_dim
        self.ord = ordinal
        lrp, lrt = learning_rate

        # set parameters
        self.activ_fn = tf.nn.tanh
        self.layer_norm = layer_norm

        # define layers
        if layers is None:
            layers = [64, 64]
        self.layers = layers

        # evaluation and target scope
        ep_scope = 'eval_policy_' + str(ordinal)
        tp_scope = 'target_policy_' + str(ordinal)
        te_scope = 'termination_' + str(ordinal)

        # Termination function
        self.p = self._termination_net(scope=te_scope, trainable=True)

        # some placeholders
        self.s = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')
        self.qg = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='q_gradient')
        self.adv = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='advantage')

        # evaluation and target network
        deterministic_policy, policy, logp_pi = self._option_net(scope=ep_scope, reuse=True)
        deterministic_policy_, policy_, logp_pi_ = self._option_net(scope=tp_scope, reuse=False)

        # soft update
        ep_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ep_scope)
        tp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tp_scope)

        te_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=te_scope)

        # update target network
        self.update = [tf.assign(t, e) for t, e in zip(tp_params, ep_params)]

        # define optimizer
        pg = tf.gradients(ys=self.a, xs=ep_params, grad_ys=self.qg)
        tg = tf.gradients(ys=self.p, xs=te_params, grad_ys=-self.adv)
        self.pop = tf.train.AdamOptimizer(-lrp).apply_gradients(zip(pg, ep_params))
        self.top = tf.train.AdamOptimizer(-lrt).apply_gradients(zip(tg, te_params))

        # for pretrain termination function
        self.prob = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.diff = tf.reduce_max(tf.abs(self.prob - self.p))
        self.loss = tf.nn.l2_loss(self.prob - self.p)
        self.oop = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

        self.train_counter = 0

    def train_option(self, state_batch, q_gradient_batch, advantage_batch=None):
        """Train the policy and termination function"""

        self.sess.run(self.pop, feed_dict={
            self.s: state_batch,
            self.qg: q_gradient_batch
        })

        if advantage_batch is not None:
            self.sess.run(self.top, feed_dict={
                self.s: state_batch,
                self.adv: advantage_batch
            })

        self.train_counter += 1

        if self.train_counter == 10:
            self.sess.run(self.update)
            self.train_counter = 0

    def choose_action(self, state):
        """Choose action"""

        return self.sess.run(self.a, feed_dict={
            self.s: state[np.newaxis, :]
        })[0]

    def get_term_prob(self, state):
        """Get termination probability of current state"""

        return self.sess.run(self.p, feed_dict={
            self.s: state[np.newaxis, :]
        })[0]

    def state_model(self, input, kernel_shapes, weight_shapes):
        weights1 = tf.get_variable(
            "weights1", kernel_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        weights2 = tf.get_variable(
            "weights2", kernel_shapes[1],
            initializer=tf.contrib.layers.xavier_initializer())
        weights3 = tf.get_variable(
            "weights3", kernel_shapes[2],
            initializer=tf.contrib.layers.xavier_initializer())
        weights4 = tf.get_variable(
            "weights5", weight_shapes[0],
            initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.get_variable(
            "q_bias1", weight_shapes[0][1],
            initializer=tf.constant_initializer())

        # Convolve
        conv1 = tf.nn.relu(tf.nn.conv2d(
            input, weights1, strides=[1, 4, 4, 1], padding='VALID'))
        conv2 = tf.nn.relu(tf.nn.conv2d(
            conv1, weights2, strides=[1, 2, 2, 1], padding='VALID'))
        conv3 = tf.nn.relu(tf.nn.conv2d(
            conv2, weights3, strides=[1, 1, 1, 1], padding='VALID'))

        # Flatten and Feedforward
        flattened = tf.contrib.layers.flatten(conv3)
        net = tf.nn.relu(tf.nn.xw_plus_b(flattened, weights4, bias1))

        return net

    def get_actions(self, state_batch):
        """Get target actions"""

        return self.sess.run(self.a, feed_dict={
            self.s: state_batch
        })

    def get_target_actions(self, state_batch):
        """Get target actions"""

        return self.sess.run(self.a_, feed_dict={
            self.s: state_batch
        })

    def _option_net(self, reuse=False, scope=""):
        """Generate evaluation/target option network"""

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(self.s, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(self.s)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ad, activation=None)

            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ad, activation=None)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)

        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)

        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probabilty
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)

        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def _termination_net(self, scope, trainable):
        """Generate evaluation/target option network"""

        with tf.variable_scope(scope):

            x = tf.layers.dense(self.s, 32, activation=tf.nn.relu,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense1')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch1')
            prob = tf.layers.dense(x, 1, activation=tf.nn.sigmoid,
                                   kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                   trainable=trainable, name='prob')

        return prob

    def render(self):
        """Render option and termination function"""

        fig, (ax0, ax1) = plt.subplots(1, 2)

        num = 100
        delta = 2.0 / num
        sta = -np.ones((num * num, 2)) + delta * 0.5
        u = np.zeros((num, num))
        v = np.zeros((num, num))
        p = np.zeros((num, num))

        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([j * delta, i * delta])
                sta[o] += s

        a = self.sess.run(self.a, feed_dict={
            self.s: sta
        })
        p1 = self.sess.run(self.p, feed_dict={
            self.s: sta
        })

        for i in range(num):
            for j in range(num):
                o = i * num + j
                u[i, j] = a[o, 0]
                v[i, j] = a[o, 1]
                p[i, j] = p1[o]

        V = (u * u + v * v) ** 0.5
        x = np.linspace(-1.0, 1.0, num + 1)
        ax0.streamplot(sta[:num, 0], sta[:num, 0], u, v, color=1.4-V)
        im0 = ax0.pcolor(x, x, V, cmap='jet')
        ax0.set_title('intra-policy')
        fig.colorbar(im0, ax=ax0)
        im1 = ax1.pcolor(x, x, p, cmap='jet')
        ax1.set_title('termination function')
        fig.colorbar(im1, ax=ax1)

        fig.tight_layout()
        plt.show()

    def pretrain(self):
        """Pretrain termination function(need to be designed specifically)"""

        ord = self.ord
        num = 50
        delta = 2.0 / (num - 1)
        test_state = -np.ones((num * num, 2))
        test_label = np.ones((num * num, 1))
        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([i * delta, j * delta])
                test_state[o] += s
                if ord == 0:
                    if test_state[o, 0] > 0 and test_state[o, 1] > 0:
                        test_label[o] = 0.0
                elif ord == 1:
                    if test_state[o, 0] < 0 < test_state[o, 1]:
                        test_label[o] = 0.0
                elif ord == 2:
                    if test_state[o, 0] < 0 and test_state[o, 1] < 0:
                        test_label[o] = 0.0
                elif ord == 3:
                    if test_state[o, 1] < 0 < test_state[o, 0]:
                        test_label[o] = 0.0

        bound = 1e-2
        while True:
            self.sess.run(self.oop, feed_dict={
                self.s: test_state,
                self.prob: test_label
            })
            a = self.sess.run(self.diff, feed_dict={
                self.s: test_state,
                self.prob: test_label
            })
            if a < bound:
                break