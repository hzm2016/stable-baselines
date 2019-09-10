#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of critic,
"""

import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from stable_baselines.common import tf_util

LAYER1 = 64
LAYER2 = 64

EVAL_SCOPE = 'eval_lower_critic'
TARGET_SCOPE = 'target_lower_critic'

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)


def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


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
        self.loss = tf.losses.mean_squared_error(labels=self.r + (1.0 - self.terminal) * gamma * q_, predictions=self.q)

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


class Soft_critic(object):

    def __init__(self, session, state_dim, action_dim, option_dim,
                 gamma, tau=1e-2, learning_rate=1e-3,
                 ent_coef='auto', target_entropy='auto', layers=None, layer_norm=False,
                 feature_extraction="mlp", act_fun=tf.nn.relu):
        """Initiate the critic network for normalized states and actions"""

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.ad = action_dim
        self.od = option_dim

        # define layers
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.activ_fn = act_fun
        self.target_entropy = target_entropy

        # some placeholder
        self.s = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')
        self.a = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='action')
        self.s_ = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='next_state')
        self.r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='reward')
        self.terminal = tf.placeholder(tf.float32, shape=(None, 1), name='terminal')
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.policy = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='policy')
        self.logp_pi = tf.placeholder(dtype=tf.float32, shape=(None, self.ad), name='log_policy')
        self.option_term_prob = tf.placeholder(dtype=tf.float32, shape=(None, self.od), name='termination_prob')

        self.tau = tau

        # given the option and state
        self.option = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='option')
        self.options_onehot = tf.squeeze(tf.one_hot(self.option, self.od,
                                                    dtype=tf.float32), [1])

        # Normalization or other features
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.ent_coef = ent_coef
        self.log_ent_coef = None

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == 'auto':
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.ad).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if '_' in self.ent_coef:
                init_value = float(self.ent_coef.split('_')[1])
                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                initializer=np.log(init_value).astype(np.float32))
            self.ent_coef = tf.exp(self.log_ent_coef)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef = float(self.ent_coef)

        # evaluation and target network
        # sampled actions
        qf1, qf2, value_fn = self._q_net(state=self.s, scope="state_value", action=self.a,
                                         create_qf=True, create_vf=True)
        # option value function
        self.qf1_pi, self.qf2_pi, _ = self._q_net(state=self.s, scope="option_state_value", action=self.policy,
                                        create_qf=True, create_vf=False, reuse=True)
        # Create the value network
        _, _, self.value_target = self._q_net(state=self.s_, scope="target_state_value",
                                              create_qf=False, create_vf=True)

        with tf.variable_scope("option_value"):
            self.option_q_value = tf.reduce_sum(self.qf1_pi * self.options_onehot, [1])
            self.option_v_value = tf.reduce_sum(self.value_target * self.options_onehot, [1])

            self.advantage_value = tf.reduce_sum(self.option_term_prob * (self.option_q_value - self.option_v_value))

        with tf.variable_scope("policy_over_options"):
            # softmax policy
            self.optimal_option_prob = tf.nn.softmax(self.value_target)

        # soft update
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="state_value")
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_state_value")

        with tf.variable_scope("loss", reuse=False):
            """ define loss """
            # Take the min of the two Q-Values (Double-Q Learning)
            min_qf_pi = tf.minimum(self.qf1_pi, self.qf2_pi)

            # Target for Q value regression
            q_backup = tf.stop_gradient(
                self.r + (1 - self.terminal) * self.gamma * self.value_target
            )

            # Compute Q-Function loss
            # TODO: test with huber loss (it would avoid too high values)
            qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
            qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

            # Compute the entropy temperature loss
            # it is used when the entropy coefficient is learned
            ent_coef_loss, entropy_optimizer = None, None
            if not isinstance(self.ent_coef, float):
                ent_coef_loss = -tf.reduce_mean(
                    self.log_ent_coef * tf.stop_gradient(self.logp_pi + self.target_entropy))
                entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Target for value fn regression
            # We update the vf towards the min of two Q-functions in order to
            # reduce overestimation bias from function approximation error.
            v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * self.logp_pi)
            value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)
            values_losses = qf1_loss + qf2_loss + value_loss

            # Value train op
            value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            values_params = get_vars('state_value')

            # value train
            self.train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

            # Polyak averaging for target variables
            self.target_update_op = [
                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                for target, source in zip(target_params, eval_params)
            ]

            # Add entropy coefficient optimization operation if needed
            if ent_coef_loss is not None:
                self.ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)

            # Monitor losses and entropy in tensorboard
            tf.summary.scalar('qf1_loss', qf1_loss)
            tf.summary.scalar('qf2_loss', qf2_loss)
            tf.summary.scalar('value_loss', value_loss)
            tf.summary.scalar('entropy', self.entropy)
            if ent_coef_loss is not None:
                tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                tf.summary.scalar('ent_coef', self.ent_coef)

            tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate))

    def train_critic(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            next_action_batch,
            terminal_batch,
            logp_pi):
        """Train the critic network"""

        _, _, _, = self.sess.run([self.train_values_op, self.target_update_op, self.ent_coef_op], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.r: reward_batch,
            self.s_: next_state_batch,
            self.a_: next_action_batch,
            self.terminal: terminal_batch,
            self.logp_pi: logp_pi
        })

        # soft update
        # self.train_counter += 1
        #
        # if self.train_counter == 10:
        #     self.sess.run(self.update)
        #     self.train_counter = 0

    def _q_net(self, state=None, action=None, reuse=False,
               scope="values_fn", create_vf=True, create_qf=True):
        """ get v and q function"""

        # if obs is None:
        #     obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.state_model(
                state, [[8, 8, 4, 32], [4, 4, 32, 64], [3, 3, 64, 64]], [[3136, 512]])
            else:
                critics_h = tf.layers.flatten(state)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

                    # mlp full-connected layers
                    value_fn = tf.layers.dense(vf_h, self.od, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, self.od, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, self.od, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def q_value_batch(self, state_batch, action_batch):
        """Get the q batch"""

        return self.sess.run(self.qf1_pi, feed_dict={
            self.s: state_batch,
            self.a: action_batch
        })

    def advantage_batch(self, state_batch, option_term_prob):
        """Get the q batch"""

        return self.sess.run([self.advantage_value], feed_dict={
            self.s: state_batch,
            self.optimal_option_prob: option_term_prob
        })

    def predict_value(self, state, option):
        """ return option value """

        return self.sess.run([self.qf1_pi, self.option_q_value, self.option_v_value], feed_dict={
            self.s: [state],
            self.option: option
        })

    def predict_option(self, state):
        """ return options """

        return self.sess.run(self.optimal_option_prob, feed_dict={
            self.s: [state]
        })

    def state_model(self, input, kernel_shapes, weight_shapes):
        """ to cope with the input is image"""

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
        flattened = tf.layers.flatten(conv3)
        net = tf.nn.relu(tf.nn.xw_plus_b(flattened, weights4, bias1))
        return net
