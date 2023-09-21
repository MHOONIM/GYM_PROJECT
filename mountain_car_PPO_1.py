import gym
import numpy as np
import tensorflow as tf
import random


# --------------------------------------------- Deep Q-Network Class Start ---------------------------------------------
class PPO:
    def __init__(self):
        # Initialise state parameters
        self.state = []
        self.next_state = []
        # self.actions = env.action_space
        self.actions = [0, 1, 2]
        self.reward = 0
        self.reward_append = []
        self.state_dim = len(env.reset()[0])
        self.action_dim = env.action_space.n
        self.terminated = False
        self.observation_size = 200
        self.step = 0
        self.epoch = 10
        self.left_bound = -0.6
        self.right_bound = -0.4

        # Initialise Experience Replay Buffers
        self.state_obs = np.zeros([self.observation_size, self.state_dim], dtype=float)
        self.action_obs = np.zeros([self.observation_size], dtype=int)  # Discrete action space
        self.old_policy_obs = np.zeros([self.observation_size], dtype=float)
        self.reward_obs = np.zeros([self.observation_size], dtype=float)
        self.terminated_obs = np.zeros([self.observation_size], dtype=bool)
        self.next_state_obs = np.zeros([self.observation_size, self.state_dim], dtype=float)
        self.full_flag = False

        # Initialise Network
        self.actor, self.critic = self.network_creation()
        self.epsilon = 0.1
        self.update_period = 100
        self.update_count = 0
        self.actor_global_grad = 0
        self.critic_global_grad = 0

    def network_creation(self):
        # Define hyperparameters
        hidden_layers = 16
        input_dimension = len(env.reset()[0])
        output_dimension = env.action_space.n

        # Input layer
        inputs = tf.keras.layers.Input(shape=input_dimension,)
        dense_1 = tf.keras.layers.Dense(hidden_layers)(inputs)
        dense_2 = tf.keras.layers.Dense(hidden_layers)(dense_1)
        # dense_3 = tf.keras.layers.Dense(hidden_layers)(dense_2)
        # dense_4 = tf.keras.layers.Dense(hidden_layers)(dense_3)
        activation_1 = tf.keras.layers.Activation('relu')(dense_2)

        # Output layer
        policy_outputs = tf.keras.layers.Dense(output_dimension, activation='softmax')(activation_1)
        value_outputs = tf.keras.layers.Dense(1, activation='linear')(activation_1)

        # Define the model
        # Actor -- Policy Network
        policy_network = tf.keras.models.Model(inputs=inputs, outputs=policy_outputs, name='policy_network')
        policy_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1))
        # Critics -- State Value Network
        state_value_network = tf.keras.models.Model(inputs=inputs, outputs=value_outputs, name='action_value_network')
        state_value_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1))
        return policy_network, state_value_network

    def custom_reward(self):
        if self.next_state[0] > self.right_bound:
            self.right_bound = self.next_state[0]
            reward = 1
        elif self.next_state[0] < self.left_bound:
            self.left_bound = self.next_state[0]
            reward = 1
        else:
            reward = 0
        if self.next_state[0] >= 0.5:
            reward = 100
        return reward

    def network_training(self, experience_size):
        gamma = 0.99
        # Initialise batch replay buffers
        a_hat = np.zeros([experience_size], dtype=float)
        y_t = np.zeros([experience_size], dtype=float)
        slice_indices = np.zeros([experience_size, 2])
        slice_indices[:, 0] = np.arange(experience_size)

        # Prepare final state in tensor (For computation of a_hat)
        final_obs_state_tensor = tf.convert_to_tensor(np.expand_dims(self.state_obs[experience_size-1], axis=0))

        for j in range(experience_size):
            # Prepare current state in tensor
            state_obs_tensor = tf.convert_to_tensor(np.expand_dims(self.state_obs[j], axis=0))
            # Compute target V-value (y_t)
            if self.terminated_obs[j]:
                y_t[j] = self.reward_obs[j]
            else:
                next_state_expand = np.expand_dims(self.next_state_obs[j], axis=0)
                next_state_tensor = tf.convert_to_tensor(next_state_expand)
                y_t[j] = self.reward_obs[j] + gamma * self.critic(next_state_tensor).numpy()[0]

            # Compute Advantages Function
            for k in range(experience_size - 1):
                if k + j > experience_size - 1:
                    a_hat[j] = a_hat[j]
                else:
                    a_hat[j] = a_hat[j] + (gamma ** k) * self.reward_obs[k + j]
            a_hat[j] = a_hat[j] + (gamma ** (experience_size - j)) * self.critic(final_obs_state_tensor).numpy()[0]
            a_hat[j] = a_hat[j] - self.critic(state_obs_tensor).numpy()[0]

        # Prepare data before updating the gradients
        state_obs_tensor = tf.convert_to_tensor(self.state_obs[0:experience_size], dtype=tf.float32)
        old_policy_tensor = tf.convert_to_tensor(self.old_policy_obs[0:experience_size], dtype=tf.float32)
        a_hat_tensor = tf.convert_to_tensor(a_hat, dtype=tf.float32)
        y_t_tensor = tf.convert_to_tensor(y_t, dtype=tf.float32)
        slice_indices[:, 1] = self.action_obs[0:experience_size]

        # Gradient Tape
        for k in range(self.epoch):
            # *********************************** Tape_1: Updating Policy Network **************************************
            with tf.GradientTape() as tape_1:
                tape_1.watch(state_obs_tensor)
                tape_1.watch(old_policy_tensor)
                tape_1.watch(a_hat_tensor)
                current_policy_tensor = self.actor(state_obs_tensor)
                current_policy_tensor = tf.gather_nd(current_policy_tensor, indices=slice_indices.astype(int))
                # Probability Ratio
                prob_ratio = tf.divide(current_policy_tensor, old_policy_tensor)
                # Unclipped
                un_clipped = prob_ratio * a_hat_tensor
                # Clipped
                clipped = tf.clip_by_value(prob_ratio,
                                           1 - self.epsilon,
                                           1 + self.epsilon)
                clipped = clipped * a_hat_tensor
                # Cost Function
                policy_cost = tf.reduce_mean(tf.minimum(un_clipped, clipped))
            policy_cost_gradient = tape_1.gradient(policy_cost, self.actor.trainable_variables)
            policy_glob_gradient = tf.linalg.global_norm(policy_cost_gradient)
            self.actor_global_grad = policy_glob_gradient.numpy()
            # Apply Gradients
            self.actor.optimizer.apply_gradients(zip(policy_cost_gradient, self.actor.trainable_variables))

            # ********************************* Tape_2: Updating Critic ************************************************
            with tf.GradientTape() as tape_2:
                tape_2.watch(state_obs_tensor)
                tape_2.watch(y_t_tensor)
                y_pred_tensor = self.critic(state_obs_tensor)
                state_value_cost = tf.keras.losses.MSE(y_t_tensor, y_pred_tensor)
            critic_cost_gradient = tape_2.gradient(state_value_cost, self.critic.trainable_variables)
            critic_glob_gradient = tf.linalg.global_norm(critic_cost_gradient)
            self.critic_global_grad = critic_glob_gradient.numpy()
            # Apply Gradients
            self.critic.optimizer.apply_gradients(zip(critic_cost_gradient, self.critic.trainable_variables))

    def agent_training(self, episode):
        for i in range(episode):
            # Reset parameters
            self.terminated = False
            step_count = 0
            sum_reward = 0
            self.left_bound = -0.6
            self.right_bound = -0.4

            # Get the initial state
            self.state = env.reset()  # 1 = Do nothing
            self.state = self.state[0]

            while not self.terminated:
                state_expand = np.expand_dims(self.state, axis=0)
                state_tensor = tf.convert_to_tensor(state_expand)
                policy = self.actor(state_tensor)
                selected_action = np.random.choice(self.actions, p=policy.numpy()[0])
                selected_action_policy = tf.gather_nd(policy, indices=(0, selected_action))

                # Interact with the environment
                self.next_state, self.reward, self.terminated, truncated, _ = env.step(selected_action)

                # Custom Reward
                self.reward = self.reward + self.custom_reward()

                # Store transitions in the Experience Replay Buffers
                state_reshape = np.reshape(self.state, (1, 2))
                next_state_reshape = np.reshape(self.next_state, (1, 2))
                self.state_obs[step_count] = state_reshape
                self.action_obs[step_count] = selected_action
                self.old_policy_obs[step_count] = selected_action_policy.numpy()
                self.reward_obs[step_count] = self.reward
                self.terminated_obs[step_count] = self.terminated
                self.next_state_obs[step_count] = next_state_reshape

                # Step increment
                self.step += 1
                step_count += 1

                # Update state if the episode is still not terminated.
                if step_count > (self.observation_size - 1):
                    self.terminated = True
                else:
                    self.state = self.next_state
                    sum_reward = sum_reward + self.reward

            # Train the model
            self.network_training(step_count)

            # Save the reward
            self.reward_append.append(sum_reward)

            # Print the status of learning
            print('episode: ', i, ', episode_reward: ', sum_reward,
                  ', avg_reward: ', np.sum(self.reward_append)/len(self.reward_append),
                  ', Actor Grad: ', self.actor_global_grad, ', Critic Grad: ', self.critic_global_grad,
                  ', last_state: ', self.state)
# ---------------------------------------------- Deep Q-Network Class End ----------------------------------------------


# ------------------------------------------------- Main Program Start -------------------------------------------------
if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = PPO()
    agent.agent_training(5000)
    agent.actor.summary()
# -------------------------------------------------- Main Program End --------------------------------------------------

