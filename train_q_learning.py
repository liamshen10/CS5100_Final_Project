# train_q_learning.py

import numpy as np
import tensorflow as tf
from q_network import QNetwork
from nfl_environment import NFLPlayPredictionEnvironment
import random

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount factor for future rewards
num_episodes = 10  # Total number of episodes for training
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration probability
epsilon_decay = 0.995  # Decay rate for exploration probability

# Initialize environment and Q-Network
env = NFLPlayPredictionEnvironment('processed_pbp_2023.csv', 'label_mappings.json')
state_size = env.state_space.shape[1]
action_size = len(env.action_space)
q_network = QNetwork(state_size, action_size)

# Optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

def reshape_state(state):
    return np.array(state).reshape(1, -1, 1)

# Metrics
cumulative_rewards = []
losses = []
epsilon_values = []

for episode in range(num_episodes):
    state = reshape_state(env.reset())
    done = False
    total_reward = 0
    episode_losses = []

    while not done:
        if random.random() < epsilon:
            action = random.choice(range(action_size))
        else:
            action = np.argmax(q_network.model.predict(state)[0])
        
        next_state, reward, done = env.step(action)

        if not done:
            next_state_reshaped = reshape_state(next_state)
            target_q = reward + gamma * np.max(q_network.model.predict(next_state_reshaped)[0])
        else:
            # End of episode: no next state, reward is final
            target_q = reward

        with tf.GradientTape() as tape:
            q_values = q_network.model(state)
            q_value = q_values[0][action]

            target_q_reshaped = tf.reshape(target_q, (1,))
            q_value_reshaped = tf.reshape(q_value, (1,))

            loss = tf.keras.losses.mean_squared_error(target_q_reshaped, q_value_reshaped)

        gradients = tape.gradient(loss, q_network.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.model.trainable_variables))

        episode_losses.append(loss.numpy())
        state = next_state_reshaped if not done else state

        total_reward += reward

    cumulative_rewards.append(total_reward)
    losses.append(np.mean(episode_losses))
    epsilon_values.append(epsilon)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Avg Loss: {np.mean(episode_losses)}, Epsilon: {epsilon}")

q_network.model.save('q_learning_nfl_model.h5')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(cumulative_rewards)
plt.title('Total Cumulative Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(3, 1, 2)
plt.plot(losses)
plt.title('Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.subplot(3, 1, 3)
plt.plot(epsilon_values)
plt.title('Epsilon Values Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()
