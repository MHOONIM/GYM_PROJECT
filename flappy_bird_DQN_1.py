import gym
import flappy_bird_gymnasium
import numpy as np
import tensorflow as tf
import random


# --------------------------------------------- Deep Q-Network Class Start ---------------------------------------------
class DQN:
    def __init__(self):
        # Initialise state parameters
        self.state = []
        self.next_state = []
        self.actions = env.action_space
        self.reward = 0
        self.reward_append = []
        self.state_dim = len(env.reset()[0])
        self.action_dim = env.action_space.n
        self.terminated = False
        self.step = 0
        self.buffer_size = 100000
        self.batch_buffer_size = 128
        self.max_step = 128
        self.epoch = 10
        self.left_bound = -0.4
        self.right_bound = -0.6

        # Initialise Experience Replay Buffers
        self.state_replay = np.zeros([self.buffer_size, self.state_dim], dtype=float)
        self.action_replay = np.zeros([self.buffer_size], dtype=int)  # Discrete action space
        self.reward_replay = np.zeros([self.buffer_size], dtype=float)
        self.terminated_replay = np.zeros([self.buffer_size], dtype=bool)
        self.next_state_replay = np.zeros([self.buffer_size, self.state_dim], dtype=float)
        self.full_flag = False

        # Initialise Network
        self.q_predict = self.network_creation()
        self.q_target = self.network_creation()
        self.q_target.set_weights(self.q_predict.get_weights())
        self.update_period = 100
        self.update_count = 0
        self.global_grad = 0

    def network_creation(self):
        # Define hyperparameters
        hidden_layers = 32
        input_dimension = len(env.reset()[0])
        output_dimension = env.action_space.n

        # Input layer
        inputs = tf.keras.layers.Input(shape=input_dimension,)
        dense_1 = tf.keras.layers.Dense(hidden_layers)(inputs)
        dense_2 = tf.keras.layers.Dense(hidden_layers)(dense_1)
        dense_3 = tf.keras.layers.Dense(hidden_layers)(dense_2)
        dense_4 = tf.keras.layers.Dense(hidden_layers)(dense_3)
        activation_1 = tf.keras.layers.Activation('relu')(dense_4)

        # Output layer
        outputs = tf.keras.layers.Dense(output_dimension, activation='linear')(activation_1)

        # Define the model
        action_value_network = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='action_value_network')
        action_value_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1))
        return action_value_network

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

    def network_training(self, indices):
        gamma = 0.99
        # Initialise batch replay buffers
        state_replay_batch = np.zeros([self.batch_buffer_size, self.state_dim], dtype=float)
        action_replay_batch = np.zeros([self.batch_buffer_size], dtype=int)
        y_t = np.zeros([self.batch_buffer_size], dtype=float)
        slice_indices = np.zeros([self.batch_buffer_size, 2])
        slice_indices[:, 0] = np.arange(self.batch_buffer_size)

        for j in range(self.batch_buffer_size):
            # Store the sampling in batch buffers
            state_replay_batch[j] = self.state_replay[indices[j]]
            action_replay_batch[j] = self.action_replay[indices[j]]

            # Compute target Q-value (y_t)
            if self.terminated_replay[indices[j]]:
                y_t[j] = self.reward_replay[indices[j]]
            else:
                next_state_replay_expand = np.expand_dims(self.next_state_replay[indices[j]], axis=0)
                next_state_replay_tensor = tf.convert_to_tensor(next_state_replay_expand)
                y_t[j] = self.reward_replay[indices[j]] + gamma * np.max(self.q_target(next_state_replay_tensor).numpy()[0])

        # Prepare data before updating the gradients
        state_replay_batch_tensor = tf.convert_to_tensor(state_replay_batch, dtype=tf.float32)
        y_t_tensor = tf.convert_to_tensor(y_t, dtype=tf.float32)
        slice_indices[:, 1] = action_replay_batch

        # Gradient Tape
        for k in range(self.epoch):
            with tf.GradientTape() as tape:
                tape.watch(state_replay_batch_tensor)
                tape.watch(y_t_tensor)
                y_predicted_tensor = self.q_predict(state_replay_batch_tensor)
                cost = tf.keras.losses.MSE(y_t_tensor,
                                           tf.gather_nd(y_predicted_tensor, indices=slice_indices.astype(int)))
                # y_predicted_tensor = tf.reduce_mean(y_predicted_tensor)
                # cost = tf.keras.losses.MSE(y_t_tensor, y_predicted_tensor)
            cost_gradient = tape.gradient(cost, self.q_predict.trainable_variables)
            glob_gradient = tf.linalg.global_norm(cost_gradient)
            self.global_grad = glob_gradient.numpy()
            # Apply Gradients
            self.q_predict.optimizer.apply_gradients(zip(cost_gradient, self.q_predict.trainable_variables))

        # Update the target network
        self.update_count += 1
        if self.update_count >= self.update_period:
            new_weights = self.q_predict.get_weights()
            self.q_target.set_weights(new_weights)
            self.update_count = 0

    def agent_training(self, episode):
        epsilon = 1  # Initial epsilon value = 1
        for i in range(episode):
            # Reset parameters
            self.terminated = False
            truncated = False
            fit = False
            step_count = 0
            sum_reward = 0

            # Get the initial state
            self.state = env.reset()  # 1 = Do nothing
            self.state = self.state[0]

            while not self.terminated:
                # Epsilon greedy policy
                rand = random.random()
                if rand < epsilon:
                    # Random action
                    selected_action = self.actions.sample()
                else:
                    # Get the action from maxQ(s, a)
                    state_expand = np.expand_dims(self.state, axis=0)
                    state_tensor = tf.convert_to_tensor(state_expand)
                    q_predicted = self.q_predict(state_tensor)
                    max_q = np.unravel_index(np.argmax(q_predicted.numpy()[0]), q_predicted.shape)
                    selected_action = max_q[1]

                # Interact with the environment
                self.next_state, self.reward, self.terminated, truncated, _ = env.step(selected_action)

                # The default reward cannot be used for the DQN, therefore, the customised reward is applied instead.
                # self.reward = self.reward + self.custom_reward()

                # Check if the buffers are full or not?
                if self.step > self.buffer_size - 1:
                    self.step = 0
                    self.full_flag = True

                # Store transitions in the Experience Replay Buffers
                state_reshape = np.reshape(self.state, (1, 4))
                next_state_reshape = np.reshape(self.next_state, (1, 4))
                self.state_replay[self.step] = state_reshape
                self.action_replay[self.step] = selected_action
                self.reward_replay[self.step] = self.reward
                self.terminated_replay[self.step] = self.terminated
                self.next_state_replay[self.step] = next_state_reshape

                # Step increment
                self.step += 1
                step_count += 1

                # Check whether the agent had been taking action for self.max_step already ?
                if self.step >= self.max_step or self.full_flag:
                    fit = True

                # Update state if the episode is still not terminated.
                if truncated:
                    self.terminated = True
                else:
                    self.state = self.next_state
                    sum_reward = sum_reward + self.reward

            # Train the model
            if fit:
                if not self.full_flag:
                    random_indices = np.arange(self.step)
                    np.random.shuffle(random_indices)
                    self.network_training(random_indices)
                else:
                    random_indices = np.arange(self.buffer_size)
                    np.random.shuffle(random_indices)
                    self.network_training(random_indices)

            # Save the reward
            self.reward_append.append(sum_reward)

            # Adaptive Epsilon
            epsilon -= 0.005
            if epsilon < 0.1:
                epsilon = 0.1

            # Print the status of learning
            print('episode: ', i, ', episode_reward: ', sum_reward,
                  ', avg_reward: ', np.sum(self.reward_append)/len(self.reward_append),
                  ', Gradient: ', self.global_grad, ', last_state: ', self.state)
# ---------------------------------------------- Deep Q-Network Class End ----------------------------------------------


# ------------------------------------------------- Main Program Start -------------------------------------------------
if __name__ == "__main__":
    env = gym.make("FlappyBird-v0", render_mode="human")
    agent = DQN()
    agent.agent_training(1000)
    agent.q_predict.summary()
# -------------------------------------------------- Main Program End --------------------------------------------------

