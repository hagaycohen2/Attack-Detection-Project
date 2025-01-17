# Authors: Hagay Cohen - 206846180 , Imri Shai - 213023500

import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import argparse

def load_data(test_data_path, le_main_path, le_secondary_path, scaler_path):
    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Separate features and labels
    X_test = test_data.iloc[:, :-2].values
    y_test_main = test_data['main_class'].values
    y_test_secondary = test_data['second_class'].values

    # Load encoders and scaler
    le_main = joblib.load(le_main_path)
    le_secondary = joblib.load(le_secondary_path)
    scaler = joblib.load(scaler_path)

    # Encode labels
    y_test_main = le_main.transform(y_test_main)
    y_test_secondary = le_secondary.transform(y_test_secondary)

    # Ensure labels are integers
    y_test_main = y_test_main.astype(int)
    y_test_secondary = y_test_secondary.astype(int)

    # Scale features
    X_test = scaler.transform(X_test)

    return X_test, y_test_main, y_test_secondary

def test(main_model, secondary_model, X_test, y_test_main, y_test_secondary):
    # Predict the test data
    main_y_pred = main_model.predict(X_test)
    secondary_y_pred = secondary_model.predict(X_test)

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

def main():
    parser = argparse.ArgumentParser(description="Data augmentation - Hallucinator")
    parser.add_argument('-t', '--test_path', dest='test_path', help='Path to test data', required=True)
    parser.add_argument('-s', '--save_path', dest='save_path', help='Path to save model', required=True)
    args = parser.parse_args()
    
    test_path = args.test_path
    save_path = args.save_path

    # Define paths
    main_model_path = os.path.join(save_path, 'Hallucinator_main.keras')
    secondary_model_path = os.path.join(save_path, 'Hallucinator_secondary.keras')
    test_data_path = test_path
    le_main_path = os.path.join(save_path, 'le_main.pkl')
    le_secondary_path = os.path.join(save_path, 'le_secondary.pkl')
    scaler_path = os.path.join(save_path, 'scaler.pkl')

    # Load models
    main_model = load_model(main_model_path)
    secondary_model = load_model(secondary_model_path)

    # Load data
    X_test, y_test_main, y_test_secondary = load_data(test_data_path, le_main_path, le_secondary_path, scaler_path)

    # Test models
    test(main_model, secondary_model, X_test, y_test_main, y_test_secondary)

if __name__ == '__main__':
    main()

# Example usage:
# python C:\Users\tnrha\Attack-Detection-Project\models\inferences\Hallucinator_inference.py -t C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv -s C:\Users\tnrha\Attack-Detection-Project\models\saved_models\Hallucinator\
