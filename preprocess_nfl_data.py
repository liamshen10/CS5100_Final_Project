import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

# Load the dataset
df = pd.read_csv('pbp-2023.csv')

# Ensure 'ToGo' is 1 or greater
df = df[df['ToGo'] >= 1]

# Ensure 'Down' is between 1 and 4
df = df[df['Down'].isin([1, 2, 3, 4])]

# Ensure 'YardLine' is between 1 and 100
df = df[(df['YardLine'] >= 1) & (df['YardLine'] <= 100)]

# Combine 'Quarter', 'Minute', and 'Second' into a single time value
df['AdjustedQuarter'] = df['Quarter'].apply(lambda x: (x - 1) * 15)
df['Time'] = (df['AdjustedQuarter'] + df['Minute']) * 60 + df['Second']

# Select columns for preprocessing
columns_to_keep = ['ToGo', 'Down', 'OffenseTeam', 'DefenseTeam', 'YardLine', 'Time', 'PlayType']
df = df[columns_to_keep]

# Initialize label encoders for categorical columns
label_encoders = {col: LabelEncoder() for col in ['OffenseTeam', 'DefenseTeam', 'PlayType']}

# Dictionary to store mappings
mappings = {}

# Apply label encoding and save mappings
for col, encoder in label_encoders.items():
    df[col] = encoder.fit_transform(df[col].astype(str))
    mappings[col] = {index: label for index, label in enumerate(encoder.classes_)}

# Save mappings to a JSON file for future reference
with open('label_mappings.json', 'w') as file:
    json.dump(mappings, file, indent=4)

# Output the processed data to a CSV file
df.to_csv('processed_pbp_2023.csv', index=False)

# Print the first few rows of the processed dataframe
print(df.head())
