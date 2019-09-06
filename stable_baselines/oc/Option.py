#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of deterministic option, including deterministic policy
and termination function, which are parameterized by different parameters.

"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)

# the maximum times to double the weights
DOUBLE_TIME = 10
LAYER1 = 64
LAYER2 = 64


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


class Option_stochastic(object):
