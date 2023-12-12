# q_network.py

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np

class QNetwork:
    def __init__(self, state_size, action_size):
        self.model = self.build_model(state_size, action_size)

    def build_model(self, state_size, action_size):
        # Neural network for Q-function approximation
        model = Sequential()

        # Adjust these layers and units according to your specific task and data
        model.add(LSTM(64, input_shape=(state_size, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))  # Output layer: one neuron per action

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def summary(self):
        # Print a summary of the neural network
        return self.model.summary()

    def test_run(self, input_shape):
        # Generate a dummy input and run a test prediction
        dummy_input = np.random.random(input_shape)
        return self.model.predict(dummy_input)

# Load the preprocessed data to determine the state and action sizes
data = pd.read_csv('processed_pbp_2023.csv')
state_size = data.shape[1] - 1  # Assuming last column is the action
action_size = len(pd.read_json('label_mappings.json')['PlayType'].to_dict())

# Create the Q-Network
q_network = QNetwork(state_size, action_size)

# Print the model summary
print("Model Summary:")
q_network.summary()

# Test run with dummy input
print("\nTest Run Output:")
dummy_input_shape = (1, state_size, 1)  # Assuming batch size of 1
print(q_network.test_run(dummy_input_shape))
