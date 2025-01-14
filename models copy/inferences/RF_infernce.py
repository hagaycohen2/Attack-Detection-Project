import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Directory containing the models
models_dir = r'C:\Users\User\Attack-Detection-Project\models copy\saved_models\RF'

# Load test data
test_data_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv'
test_data = pd.read_csv(test_data_path)
X_test = test_data.iloc[:, :-2].values
y_test_main = test_data['main_class'].values
y_test_secondary = test_data['second_class'].values


# List of model filenames
main_model = joblib.load(os.path.join(models_dir, 'rf_main.pkl'))
secondary_model = joblib.load(os.path.join(models_dir, 'rf_second.pkl'))

# Predict test data
main_predictions = main_model.predict(X_test)
secondary_predictions = secondary_model.predict(X_test)

# Calculate and print classification reports
main_report = classification_report(y_test_main, main_predictions)
print("Main Class Classification Report:")
print(main_report)

secondary_report = classification_report(y_test_secondary, secondary_predictions)
print("Secondary Class Classification Report:")
print(secondary_report)

# Calculate and print accuracies
main_accuracy = accuracy_score(y_test_main, main_predictions)
print("Main Class Accuracy:", main_accuracy)

secondary_accuracy = accuracy_score(y_test_secondary, secondary_predictions)
print("Secondary Class Accuracy:", secondary_accuracy)

# overall accuracy
overall_accuracy = (main_accuracy + secondary_accuracy) / 2
print("Overall Accuracy:", overall_accuracy)