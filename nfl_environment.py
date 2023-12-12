# nfl_environment.py

import pandas as pd
import json
import random  # Add this import statement

class NFLPlayPredictionEnvironment:
    def __init__(self, data_path, mapping_path):
        # Load preprocessed data
        self.data = pd.read_csv(data_path)

        # Load label mappings
        with open(mapping_path, 'r') as file:
            self.mappings = json.load(file)

        # Define state and action spaces
        self.state_space = self.define_state_space()
        self.action_space = self.define_action_space()

    def define_state_space(self):
        # State space includes 'Down', 'ToGo', 'YardLine', 'Time', 'OffenseTeam', and 'DefenseTeam'
        state_features = ['Down', 'ToGo', 'YardLine', 'Time', 'OffenseTeam', 'DefenseTeam']
        state_space = self.data[state_features].values
        return state_space

    def define_action_space(self):
        # Action space is derived from 'PlayType'
        action_space = self.data['PlayType'].unique()
        return action_space

    def get_state_action_size(self):
        # Return the size of the state and action space
        return self.state_space.shape[1], len(self.action_space)

    def get_sample(self, index):
        # Return a sample from the dataset (state, action)
        state = self.state_space[index]
        action = self.data.loc[index, 'PlayType']
        return state, action

    def reset(self):
        # Reset the environment to the initial state
        initial_index = random.randint(0, len(self.state_space) - 1)
        self.current_state_index = initial_index
        return self.state_space[initial_index]


    def step(self, action):
        # Increment the state index
        self.current_state_index += 1

        # Check if the index is out of bounds
        if self.current_state_index >= len(self.data):
            done = True
            reward = 0  # Or some default reward
            next_state = None  # Or some default state
        else:
            done = False
            # Get the next state using get_sample and calculate the reward
            next_state, _ = self.get_sample(self.current_state_index)
            reward = self.calculate_reward(action, self.current_state_index)

        return next_state, reward, done


    def calculate_reward(self, action, state_index):
        # Define how rewards are calculated
        # Placeholder logic: reward is 1 if the action matches the dataset, 0 otherwise
        correct_action = self.data.loc[state_index, 'PlayType']
        return 1 if action == correct_action else 0

# File paths
data_path = 'processed_pbp_2023.csv'
mapping_path = 'label_mappings.json'

# Create the environment
env = NFLPlayPredictionEnvironment(data_path, mapping_path)

# Example usage
state_size, action_size = env.get_state_action_size()
print("State size:", state_size)
print("Action size:", action_size)

# Displaying the first 5 samples of state-action pairs
print("\nFirst 5 samples of state-action pairs:")
for i in range(5):
    state, action = env.get_sample(i)
    print(f"Sample {i + 1} - State: {state}, Action: {action}")

# Displaying the action space mapping
print("\nAction space mapping (PlayType):")
for key, value in env.mappings['PlayType'].items():
    print(f"{key}: {value}")
