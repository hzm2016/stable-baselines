#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The upper policy and the whole training process defined here
"""
import numpy as np
import tensorflow as tf


from algorithms.Option import Option
from algorithms.LCritic import LCritic
from algorithms.Buffer import Buffer
from algorithms.UCritic import UCritic

# the number of models that will be saved
MODEL_NUM = 100
DELAY = 10000


class Brain:

    def __init__(self, env_dict, params):
        """
        option_num, state_dim, action_dim, action_bound, gamma, learning_rate, replacement,
                 buffer_capacity, epsilon
        gamma: (u_gamma, l_gamma)
        learning_rate: (lr_u_policy, lr_u_critic, lr_option, lr_termin, lr_l_critic)
        """

        # session
        self.sess = tf.Session()

        # environment parameters
        self.sd = env_dict['state_dim']
        self.ad = env_dict['action_dim']
        a_bound = env_dict['action_scale']
        assert a_bound.shape == (self.ad,), 'Action bound does not match action dimension!'

        # hyper parameters
        self.on = params['option_num']
        epsilon = params['epsilon']
        u_gamma = params['upper_gamma']
        l_gamma = params['lower_gamma']
        u_capac = params['upper_capacity']
        l_capac = params['lower_capacity']
        u_lrcri = params['upper_learning_rate_critic']
        l_lrcri = params['lower_learning_rate_critic']
        l_lrpol = params['lower_learning_rate_policy']
        l_lrter = params['lower_learning_rate_termin']

        # Upper critic and buffer
        self.u_critic = UCritic(session=self.sess, state_dim=self.sd, option_num=self.on,
                                gamma=u_gamma, epsilon=epsilon, learning_rate=u_lrcri)
        self.u_buffer = Buffer(state_dim=self.sd, action_dim=1, capacity=u_capac)

        # Lower critic, options and buffer HER
        self.l_critic = LCritic(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                gamma=l_gamma, learning_rate=l_lrcri)

        """options and buffers"""
        self.l_options = [Option(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                 ordinal=i, learning_rate=[l_lrpol, l_lrter])
                          for i in range(self.on)]
        self.l_buffers = [Buffer(state_dim=self.sd, action_dim=self.ad, capacity=l_capac)
                          for i in range(self.on)]

        # Initialize all coefficients and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=MODEL_NUM)

        # counter for training termination
        self.tc = 0

    def train_policy(self, batch_size):
        """Train upper critic(policy)"""

        if self.u_buffer.pointer > batch_size:

            # sample batches
            state_batch, option_batch, reward_batch, next_state_batch, _ = self.u_buffer.sample(batch_size)

            # training
            self.u_critic.train(state_batch, option_batch, reward_batch, next_state_batch)

    def train_option(self, batch_size, option):
        """Train option and l_critic"""

        if self.l_buffers[option].pointer > batch_size:

            # sample batches
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
                self.l_buffers[option].sample(batch_size)

            next_action_batch = self.l_options[option].get_target_actions(next_state_batch)

            # train lower critic
            self.l_critic.train(state_batch, action_batch, reward_batch, next_state_batch,
                                next_action_batch, terminal_batch)

            # get affiliated batch
            q_gradients_batch = self.l_critic.q_gradients(state_batch, action_batch)

            if self.tc == DELAY:
                advantage_batch = self.l_critic.q_batch(state_batch, action_batch) - \
                                  self._value_batch(state_batch)
                self.tc = 0
            else:
                advantage_batch = None
                self.tc += 1

            # train lower options
            self.l_options[option].train(state_batch, q_gradients_batch, advantage_batch)

            return True

        return False

    def save_model(self, model_name, step):
        """Save current model"""

        self.saver.save(self.sess, './model/' + model_name + '.ckpt', global_step=step, write_meta_graph=True)

    def restore_model(self, model_name):
        """Restore trained model"""

        ckpt = tf.train.get_checkpoint_state('./model/' + model_name)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def _value_batch(self, state_batch):
        """The upper policy average of Q value for each option
        :return: the value
        """

        batch_size = state_batch.shape[0]
        value_batch = np.zeros((batch_size, 1))
        action_batch = [self.l_options[i].get_actions(state_batch) for i in range(self.on)]
        q_batch = [self.l_critic.q_batch(state_batch, action_batch[i]) for i in range(self.on)]
        distribution_batch = self.u_critic.get_distribution(state_batch)

        # calculate the value function
        for i in range(batch_size):
            for j in range(self.on):
                value_batch[i] += q_batch[j][i] * distribution_batch[i, j]

        return value_batch


import sys
import time
import multiprocessing
from collections import deque
import warnings

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger

class soft_option_critic(object):
    def __init__(self, env_dict, params):
        """
        option_num, state_dim, action_dim, action_bound, gamma, learning_rate, replacement,
                 buffer_capacity, epsilon
        gamma: (u_gamma, l_gamma)
        learning_rate: (lr_u_policy, lr_u_critic, lr_option, lr_termin, lr_l_critic)
        """

        # session
        self.sess = tf.Session()

        # environment parameters
        self.sd = env_dict['state_dim']
        self.ad = env_dict['action_dim']
        a_bound = env_dict['action_scale']
        assert a_bound.shape == (self.ad,), 'Action bound does not match action dimension!'

        # hyper parameters
        self.on = params['option_num']
        epsilon = params['epsilon']
        u_gamma = params['upper_gamma']
        l_gamma = params['lower_gamma']
        u_capac = params['upper_capacity']
        l_capac = params['lower_capacity']
        u_lrcri = params['upper_learning_rate_critic']
        l_lrcri = params['lower_learning_rate_critic']
        l_lrpol = params['lower_learning_rate_policy']
        l_lrter = params['lower_learning_rate_termin']

        # Upper critic and buffer
        self.u_critic = UCritic(session=self.sess, state_dim=self.sd, option_num=self.on,
                                gamma=u_gamma, epsilon=epsilon, learning_rate=u_lrcri)
        self.u_buffer = Buffer(state_dim=self.sd, action_dim=1, capacity=u_capac)

        # Lower critic, options and buffer HER
        self.l_critic = LCritic(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                gamma=l_gamma, learning_rate=l_lrcri)

        # options and buffers
        self.l_options = [Option(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                 ordinal=i, learning_rate=[l_lrpol, l_lrter])
                          for i in range(self.on)]
        self.l_buffers = [Buffer(state_dim=self.sd, action_dim=self.ad, capacity=l_capac)
                          for i in range(self.on)]

        # Initialize all coefficients and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=MODEL_NUM)

        # counter for training termination
        self.tc = 0

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2
            self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

            with tf.variable_scope("input", reuse=False):
                # Create policy and target TF objects
                self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                             **self.policy_kwargs)

                self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)

                # Initialize Placeholders
                self.observations_ph = self.policy_tf.obs_ph

                # Normalized observation for pixels
                self.processed_obs_ph = self.policy_tf.processed_obs
                self.next_observations_ph = self.target_policy.obs_ph
                self.processed_next_obs_ph = self.target_policy.processed_obs
                self.action_target = self.target_policy.action_ph
                self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                 name='actions')
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

            with tf.variable_scope("model", reuse=False):
                # Create the policy
                # first return value corresponds to deterministic actions
                # policy_out corresponds to stochastic actions, used for training
                # logp_pi is the log probabilty of actions taken by the policy
                self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)

                # Monitor the entropy of the policy,
                # this is not used for training
                self.entropy = tf.reduce_mean(self.policy_tf.entropy)

                #  Use two Q-functions to improve performance by reducing overestimation bias.
                qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                 create_qf=True, create_vf=True)
                qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                policy_out, create_qf=True, create_vf=False,
                                                                reuse=True)

                # Target entropy is used when learning the entropy coefficient
                if self.target_entropy == 'auto':
                    # automatically set target entropy if needed
                    self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                else:
                    # Force conversion
                    # this will also throw an error for unexpected string
                    self.target_entropy = float(self.target_entropy)

                # The entropy coefficient or entropy can be learned automatically
                # see Automating Entropy Adjustment for Maximum Entropy RL section
                # of https://arxiv.org/abs/1812.05905
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

            with tf.variable_scope("target", reuse=False):
                # Create the value network
                _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                     create_qf=False, create_vf=True)
                self.value_target = value_target

            with tf.variable_scope("loss", reuse=False):
                # Take the min of the two Q-Values (Double-Q Learning)
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                # Target for Q value regression
                q_backup = tf.stop_gradient(
                    self.rewards_ph + (1 - self.terminals_ph) * self.gamma * self.value_target
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
                        self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                    entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                # Compute the policy loss
                # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                # NOTE: in the original implementation, they have an additional
                # regularization loss for the gaussian parameters
                # this is not used for now
                # policy_loss = (policy_kl_loss + policy_regularization_loss)
                policy_loss = policy_kl_loss

                # Target for value fn regression
                # We update the vf towards the min of two Q-functions in order to
                # reduce overestimation bias from function approximation error.
                v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                values_losses = qf1_loss + qf2_loss + value_loss

                # Policy train op
                # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                # Value train op
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                values_params = get_vars('model/values_fn')

                source_params = get_vars("model/values_fn/vf")
                target_params = get_vars("target/values_fn/vf")

                # Polyak averaging for target variables
                self.target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]
                # Initializing target to match source variables
                target_init_op = [
                    tf.assign(target, source)
                    for target, source in zip(target_params, source_params)
                ]

                # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                # and we first need to compute the policy action before computing q values losses
                with tf.control_dependencies([policy_train_op]):
                    train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                    self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                    # All ops to call during one training step
                    self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                     value_loss, qf1, qf2, value_fn, logp_pi,
                                     self.entropy, policy_train_op, train_values_op]

                    # Add entropy coefficient optimization operation if needed
                    if ent_coef_loss is not None:
                        with tf.control_dependencies([train_values_op]):
                            ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                            self.infos_names += ['ent_coef_loss', 'ent_coef']
                            self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                # Monitor losses and entropy in tensorboard
                tf.summary.scalar('policy_loss', policy_loss)
                tf.summary.scalar('qf1_loss', qf1_loss)
                tf.summary.scalar('qf2_loss', qf2_loss)
                tf.summary.scalar('value_loss', value_loss)
                tf.summary.scalar('entropy', self.entropy)
                if ent_coef_loss is not None:
                    tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                    tf.summary.scalar('ent_coef', self.ent_coef)

                tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

            # Retrieve parameters that must be saved
            self.params = get_vars("model")
            self.target_params = get_vars("target/values_fn/vf")

            # Initialize Variables and target network
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(target_init_op)

            self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)

            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()

            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                        or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)

                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):

        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)
