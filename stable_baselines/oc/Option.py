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
    def __init__(self, session, state_dim, action_dim, option_dim,
                 tau=1e-2, learning_rate=1e-3, target_entropy='auto',
                 ent_coef='auto', layers=None, layer_norm=False):

        """
        :param learning_rate: (learning_rate_policy, learning_rate_termin)
        :param ordinal: the name to tell different options apart
        """

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.ad = action_dim
        self.od = option_dim
        self.learning_rate_policy, self.learning_rate_termination = learning_rate

        # set parameters
        self.activ_fn = tf.nn.tanh
        self.layer_norm = layer_norm
        self.target_entropy = target_entropy
        self.tau = tau

        # define layers
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.ent_coef = ent_coef

        # evaluation and target scope
        ep_scope = 'eval_policy'
        tp_scope = 'target_policy'
        te_scope = 'termination'

        # define parameters to collect
        self.infos_names = None
        self.step_ops = None

        # Termination function
        self.termination_probs = self._termination_net(scope=te_scope, trainable=True)
        te_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=te_scope)

        # some placeholders
        self.s = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')

        # given the option and state
        self.option = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='option')

        self.options_onehot = tf.squeeze(tf.one_hot(self.option, self.od,
                                                    dtype=tf.float32), [1])

        self.option_value = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='option_gradient')
        self.adv = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='advantage_value')

        # evaluation and target network
        self.deterministic_policy, self.policy, self.logp_pi = self._option_net(scope=ep_scope, reuse=True)
        self.deterministic_policy_, self.policy_, self.logp_pi_ = self._option_net(scope=tp_scope, reuse=False)

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == 'auto':
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
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

        # Compute the entropy temperature loss
        # it is used when the entropy coefficient is learned
        ent_coef_loss, entropy_optimizer = None, None
        if not isinstance(self.ent_coef, float):
            ent_coef_loss = -tf.reduce_mean(
                self.log_ent_coef * tf.stop_gradient(self.logp_pi + self.target_entropy))
            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_policy)

        # soft update
        ep_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ep_scope)
        tp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tp_scope)

        # define loss
        with tf.variable_scope("loss", reuse=False):

            with tf.variable_scope("policy_loss", reuse=False):
                # Compute the policy loss
                # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                policy_kl_loss = tf.reduce_mean(self.ent_coef * self.logp_pi - self.option_value)

                # NOTE: in the original implementation, they have an additional
                # regularization loss for the gaussian parameters
                # this is not used for now
                # policy_loss = (policy_kl_loss + policy_regularization_loss)
                self.policy_loss = policy_kl_loss

                # Policy train op
                # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_policy)
                self.policy_train_op = policy_optimizer.minimize(self.policy_loss, var_list=ep_params)
                self.step_ops += [self.policy_train_op]

            with tf.variable_scope("termination_loss", reuse=False):
                self.term_gradient = tf.gradients(ys=self.p, xs=te_params, grad_ys=-self.adv)
                self.termination_train_op = \
                    tf.train.AdamOptimizer(-self.learning_rate_termination).apply_gradients(zip(self.term_gradient, te_params))
                self.step_ops += [self.termination_train_op]

            # Add entropy coefficient optimization operation if needed
            if ent_coef_loss is not None:
                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                self.step_ops += [ent_coef_op, self.ent_coef]

            # pretrain
            if self.pretrain:
                # for pretrain termination function
                self.prob = tf.placeholder(dtype=tf.float32, shape=(None, 1))
                self.diff = tf.reduce_max(tf.abs(self.prob - self.p))
                self.loss = tf.nn.l2_loss(self.prob - self.p)
                self.oop = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

                self.step_ops += [self.oop]

        # collect termination prob
        with tf.variable_scope("termination_probs", reuse=False):
            self.option_term_prob = tf.reduce_sum(
                self.termination_probs * self.options_onehot, [1])

        # update target network
        self.target_update_op = [
            tf.assign(target, (1 - self.tau) * target + self.tau * source)
            for target, source in zip(tp_params, ep_params)
        ]
        self.step_ops += [self.target_update_op]

        # self.update = [tf.assign(t, e) for t, e in zip(tp_params, ep_params)]
        self.train_counter = 0

    def _option_net(self, reuse=False, scope=""):
        """Generate evaluation/target option network"""

        with tf.variable_scope(scope, reuse=reuse):

            # contact the effect of option
            input_s = tf.concat([self.s, self.option], axis=1)

            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(input_s, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(input_s)

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
        self.entropy = tf.reduce_mean(gaussian_entropy(log_std))

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

            prob = tf.layers.dense(x, self.od, activation=tf.nn.sigmoid,
                                   kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                   trainable=trainable, name='prob')

        return prob

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

    def train_option(self, state_batch, q_value_batch, option_batch, advantage_batch=None):
        """Train the policy and termination function"""

        _, _, _ = self.sess.run(self.step_ops,
                      feed_dict={
                          self.s: state_batch,
                          self.option: option_batch,
                          self.option_value: q_value_batch,
                          self.adv: advantage_batch})

        # self.train_counter += 1
        #
        # if self.train_counter == 10:
        #     self.sess.run(self.update)
        #     self.train_counter = 0

    def choose_action(self, state, option):
        """Choose action"""

        return self.sess.run(self.deterministic_policy_, feed_dict={
            self.s: state[np.newaxis, :],
            self.option: option
        })[0]

    def get_term_prob(self, state, option):
        """Get termination probability of current state"""

        return self.sess.run([self.termination_probs, self.option_term_prob], feed_dict={
            self.s: state[np.newaxis, :],
            self.option: option
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