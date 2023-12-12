# decision_tree_benchmark.py

import pandas as pd
import numpy as np  # Import NumPy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
data = pd.read_csv('processed_pbp_2023.csv')

# Encoding categorical variables (if any)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == type(object):
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Splitting the dataset into features and target variable
X = data.drop('PlayType', axis=1)  # Replace 'PlayType' with the actual target column name
y = data['PlayType']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Calculate metrics
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass classification
accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Use 'ovr' for multiclass classification

# Print metrics
print(f"F1 Score: {f1}")
print(f"Test Accuracy: {accuracy}")
print(f"Test Loss: {test_loss}")
print(f"AUC Score: {auc}")

# Plotting feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
