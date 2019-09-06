#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of critic,
"""

import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

LAYER1 = 64
LAYER2 = 64

EVAL_SCOPE = 'eval_lower_critic'
TARGET_SCOPE = 'target_lower_critic'

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)


class LCritic:

    def __init__(self, session, state_dim, action_dim, gamma, tau=1e-2, learning_rate=1e-3):
        """Initiate the critic network for normalized states and actions"""

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.ad = action_dim

        # some placeholder
        self.r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='lower_reward')
        self.terminal = tf.placeholder(tf.float32, shape=(None, 1), name='terminal')

        # evaluation and target network
        self.s, self.a, self.q = self._q_net(scope=EVAL_SCOPE, trainable=True)
        self.s_, self.a_, q_ = self._q_net(scope=TARGET_SCOPE, trainable=False)

        # soft update
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=EVAL_SCOPE)
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TARGET_SCOPE)
        self.update = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]
        critic_reg_vars = [var for var in eval_params if
                           'kernel' in var.name and 'output' not in var.name]
        # define the error and optimizer
        self.loss = tf.losses.mean_squared_error(labels=self.r + (1.0 - self.terminal) * gamma * q_, predictions=self.q) + tf.
        critic_reg = tc.layers.apply_regularization(
            tc.layers.l2_regularizer(1e-2),
            weights_list=critic_reg_vars
        )
        self.loss += critic_reg
        self.op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=eval_params)

        # q gradient w.r.t. action
        self.qg = tf.gradients(ys=self.q, xs=self.a)[0]

        self.train_counter = 0

    def train(self, state_batch, action_batch, reward_batch, next_state_batch, next_action_batch, terminal_batch):
        """Train the critic network"""

        # minimize the loss
        self.sess.run(self.op, feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.r: reward_batch,
            self.s_: next_state_batch,
            self.a_: next_action_batch,
            self.terminal: terminal_batch
        })
        self.train_counter += 1

        if self.train_counter == 10:
            self.sess.run(self.update)
            self.train_counter = 0

    def q_batch(self, state_batch, action_batch):
        """Get the q batch"""

        return self.sess.run(self.q, feed_dict={
            self.s: state_batch,
            self.a: action_batch
        })

    def q_gradients(self, state_batch, action_batch):
        """Get the q gradients batch"""

        return self.sess.run(self.qg, feed_dict={
            self.s: state_batch,
            self.a: action_batch
        })

    def _q_net(self, scope, trainable):
        """Generate evaluation/target q network"""
        activation = tf.nn.tanh

        with tf.variable_scope(scope):

            state = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')
            action = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='action')

            ws = tf.get_variable(name='ws', shape=(self.sd, LAYER1), dtype=tf.float32,
                                 initializer=INIT_WEIGHT, trainable=trainable)
            wa = tf.get_variable(name='wa', shape=(self.ad, LAYER1), dtype=tf.float32,
                                 initializer=INIT_WEIGHT, trainable=trainable)
            b = tf.get_variable(name='b', shape=(1, LAYER1), dtype=tf.float32,
                                initializer=INIT_BIAS, trainable=trainable)

            """the first layer"""
            x = tf.nn.tanh(tf.matmul(state, ws) + tf.matmul(action, wa) + b, name='dense1')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch1')

            """the second layer"""
            x = tf.layers.dense(x, LAYER2, activation=activation,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense2')
            x = tf.layers.batch_normalization(x, training=True, trainable=trainable, name='batch2')

            """last layer"""
            q = tf.layers.dense(x, 1, activation=None,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='q')

        return state, action, q

