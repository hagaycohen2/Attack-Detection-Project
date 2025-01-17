# Authors: Hagay Cohen - 206846180 , Imri Shai - 213023500

import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import argparse

def load_data(test_data_path):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.iloc[:, :-2].values
    y_test_main = test_data['main_class'].values
    y_test_secondary = test_data['second_class'].values
    return X_test, y_test_main, y_test_secondary

def test(main_model, secondary_model, X_test, y_test_main, y_test_secondary):
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

def main():
    parser = argparse.ArgumentParser(description="Random Forest Inference")
    parser.add_argument('-t', '--test_path', dest='test_path', help='Path to test data', required=True)
    parser.add_argument('-s', '--save_path', dest='save_path', help='Path to save model', required=True)
    args = parser.parse_args()
    
    test_path = args.test_path
    save_path = args.save_path

    # Define paths
    models_dir = save_path
    test_data_path = test_path

    # Load models
    main_model = joblib.load(os.path.join(models_dir, 'rf_main.pkl'))
    secondary_model = joblib.load(os.path.join(models_dir, 'rf_second.pkl'))

    # Load data
    X_test, y_test_main, y_test_secondary = load_data(test_data_path)

    # Test models
    test(main_model, secondary_model, X_test, y_test_main, y_test_secondary)

if __name__ == '__main__':
    main()

# Example usage:
# python C:\Users\tnrha\Attack-Detection-Project\models\inferences\RF_infernce.py -t C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv -s C:\Users\tnrha\Attack-Detection-Project\models\saved_models\RF\