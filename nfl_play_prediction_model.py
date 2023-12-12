# nfl_play_prediction_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def create_model(input_shape, output_size):
    # Create a Sequential model
    model = Sequential()

    # Adding LSTM layers
    # Adjust the number of units and dropout according to your needs and computational capacity
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    # Adding Dense layers for output
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_size, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Assuming an example input shape and output size
# Example: input_shape = (number of timesteps, number of features)
input_shape = (10, 5)  # Modify based on your actual input data shape
output_size = 20  # Modify based on your action space size

# Create the model
model = create_model(input_shape, output_size)

# Print the model summary to confirm its structure
model.summary()
