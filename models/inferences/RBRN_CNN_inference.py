import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess test dataset
def load_test_data_as_images(test_path, save_path):
    test_data = pd.read_csv(test_path)

    # Features and labels
    X_test = test_data.iloc[:, :-2].values
    y_test_main = test_data[['main_class']].values
    y_test_secondary = test_data[['second_class']].values

    # Load encoders and scaler
    le_main = joblib.load(save_path + 'le_main.pkl')
    le_secondary = joblib.load(save_path + 'le_secondary.pkl')
    scaler = joblib.load(save_path + 'scaler.pkl')

    # Encode labels
    y_test_main = le_main.transform(y_test_main.ravel())
    y_test_secondary = le_secondary.transform(y_test_secondary.ravel())

    # Ensure labels are integers
    y_test_main = y_test_main.astype(int)
    y_test_secondary = y_test_secondary.astype(int)

    # Normalize features to [0, 1] for image transformation
    X_test = scaler.transform(X_test)

    
    return  y_test_main, y_test_secondary



 # Evaluate the model
    # test_path = r'C:\Users\user\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv'
    # evaluate_rbrn(test_path, save_path)

# Create pairs for training
def create_pairs(features, labels):
    pairs, pair_labels = [], []
    num_samples = len(features)

    # Generate positive and negative pairs
    for i in range(num_samples):
        for j in range(num_samples):
            pairs.append([features[i], features[j]])
            pair_labels.append(1 if labels[i] == labels[j] else 0)

    return np.array(pairs), np.array(pair_labels)


# Load the RBRN model and evaluate it

def evaluate_rbrn(test_path, save_path):
    # Load test data
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, :-2].values  # Features
    y_test_main = test_data[['main_class']].values
    y_test_secondary = test_data[['second_class']].values

    # Normalize features
    scaler = joblib.load(save_path + 'scaler.pkl')
    X_test = scaler.transform(X_test)

    # Load encoders
    le_main = joblib.load(save_path + 'le_main.pkl')
    le_secondary = joblib.load(save_path + 'le_secondary.pkl')

    # Encode labels
    y_test_main = le_main.transform(y_test_main.ravel())
    y_test_secondary = le_secondary.transform(y_test_secondary.ravel())

    # Ensure labels are integers
    y_test_main = y_test_main.astype(int)
    y_test_secondary = y_test_secondary.astype(int)

    # Load models
    feature_extractor = load_model(save_path + 'feature_extractor.keras')
    relation_network_main = load_model(save_path + 'relation_network_main.keras')
    relation_network_secondary = load_model(save_path + 'relation_network_secondary.keras')

    # Extract features
    test_features = feature_extractor.predict(X_test)

    # Create pairs
    test_pairs, test_pair_labels = create_pairs(test_features, y_test_main)
    test_pairs_secondary, test_pair_labels_secondary = create_pairs(test_features, y_test_secondary)

    # One-hot encode labels
    num_main_classes = len(np.unique(y_test_main))
    num_secondary_classes = len(np.unique(y_test_secondary))
    test_pair_labels = np.eye(num_main_classes)[test_pair_labels]
    test_pair_labels_secondary = np.eye(num_secondary_classes)[test_pair_labels_secondary]

    # Predict
    main_predictions = relation_network_main.predict([test_pairs[:, 0], test_pairs[:, 1]])
    secondary_predictions = relation_network_secondary.predict([test_pairs_secondary[:, 0], test_pairs_secondary[:, 1]])

    # Evaluate and print reports
    main_pred_labels = np.argmax(main_predictions, axis=1)
    secondary_pred_labels = np.argmax(secondary_predictions, axis=1)

    print("Main Class Classification Report:")
    print(classification_report(np.argmax(test_pair_labels, axis=1), main_pred_labels))

    print("Secondary Class Classification Report:")
    print(classification_report(np.argmax(test_pair_labels_secondary, axis=1), secondary_pred_labels))

    # Calculate accuracy
    main_accuracy = accuracy_score(np.argmax(test_pair_labels, axis=1), main_pred_labels)
    secondary_accuracy = accuracy_score(np.argmax(test_pair_labels_secondary, axis=1), secondary_pred_labels)

    print(f"Main Class Accuracy: {main_accuracy}")
    print(f"Secondary Class Accuracy: {secondary_accuracy}")

    # Calculate overall accuracy
    overall_accuracy = (main_accuracy + secondary_accuracy) / 2
    print(f"Overall Accuracy: {overall_accuracy}")

def main():
    test_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv'
    save_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\RBRN\\'
    evaluate_rbrn(test_path, save_path)

if __name__ == "__main__":
    main()
