import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import tensorflow as tf

# Define custom distance layer
class L1Distance(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, vectors):
        x, y = vectors
        return tf.abs(x - y)

    def get_config(self):
        config = super().get_config()
        return config

# Load the models from the saved models folder
main_model_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\siamese_network_main.keras'
secondary_model_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\siamese_network_secondary.keras'

# Define and register the custom layer (same L1Distance class as above)
tf.keras.utils.get_custom_objects()['L1Distance'] = L1Distance

main_model = load_model(main_model_path)
secondary_model = load_model(secondary_model_path)

# Load the test data
test_data_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv'
test_data = pd.read_csv(test_data_path)

# Separate features and labels
X_test = test_data.iloc[:, :-2].values
y_test_main = test_data['main_class'].values
y_test_secondary = test_data['second_class'].values

# Load encoders and scaler
le_main = joblib.load(r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\le_main.pkl')
le_secondary = joblib.load(r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\le_secondary.pkl')
scaler = joblib.load(r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\scaler.pkl')

# Encode labels
y_test_main = le_main.transform(y_test_main)
y_test_secondary = le_secondary.transform(y_test_secondary)

# Ensure labels are integers
y_test_main = y_test_main.astype(int)
y_test_secondary = y_test_secondary.astype(int)

# Scale features
X_test = scaler.transform(X_test)

# Reshape for Conv1D
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create pairs of inputs for the Siamese network
X_test_pairs = [X_test, X_test]

# Predict the test data
main_y_pred = main_model.predict(X_test_pairs)
secondary_y_pred = secondary_model.predict(X_test_pairs)

# Debugging prints
print("Main Model Predictions:", main_y_pred)
print("Secondary Model Predictions:", secondary_y_pred)
print("Main Test Labels:", y_test_main)
print("Secondary Test Labels:", y_test_secondary)

# Calculate and print the classification reports
main_report = classification_report(y_test_main, np.argmax(main_y_pred, axis=1), zero_division=0)
print("Main Class Classification Report:")
print(main_report)

secondary_report = classification_report(y_test_secondary, np.argmax(secondary_y_pred, axis=1), zero_division=0)
print("Secondary Class Classification Report:")
print(secondary_report)

# Calculate and print the accuracies
main_accuracy = accuracy_score(y_test_main, np.argmax(main_y_pred, axis=1))
print("Main Class Accuracy:", main_accuracy)

secondary_accuracy = accuracy_score(y_test_secondary, np.argmax(secondary_y_pred, axis=1))
print("Secondary Class Accuracy:", secondary_accuracy)

# Calculate and print the overall accuracy
overall_accuracy = (main_accuracy + secondary_accuracy) / 2
print("Overall Accuracy:", overall_accuracy)
