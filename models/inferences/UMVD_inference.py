import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess test dataset
def load_test_data_as_images(test_path, image_size, save_path):
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

    # Reshape into grayscale images
    X_test_images = X_test.reshape(-1, *image_size)

    return X_test_images, y_test_main, y_test_secondary

# Classification using prototypes
def classify_with_prototypes(features, prototypes):
    distances = np.linalg.norm(features[:, None, :] - prototypes[None, :, :], axis=-1)
    return np.argmin(distances, axis=1)

# Load and evaluate UMVD-FSL
def evaluate_umvd_fsl(test_path, image_size, save_path):
    # Load test data
    X_test, y_test_main, y_test_secondary = load_test_data_as_images(test_path, image_size, save_path)

    # Load feature extractor model
    feature_extractor = load_model(save_path + 'feature_extractor.h5')

    # Extract features for test images
    test_features = feature_extractor.predict(X_test)

    # Load prototypes
    main_prototypes = np.load(save_path + 'main_prototypes.npy')
    secondary_prototypes = np.load(save_path + 'secondary_prototypes.npy')

    # Classify test samples for main and secondary classes
    main_predictions = classify_with_prototypes(test_features, main_prototypes)
    secondary_predictions = classify_with_prototypes(test_features, secondary_prototypes)

    # Calculate and print the classification reports
    main_report = classification_report(y_test_main, main_predictions, zero_division=0)
    print("Main Class Classification Report:")
    print(main_report)

    secondary_report = classification_report(y_test_secondary, secondary_predictions, zero_division=0)
    print("Secondary Class Classification Report:")
    print(secondary_report)

    # Calculate and print the accuracies
    main_accuracy = accuracy_score(y_test_main, main_predictions)
    print("Main Class Accuracy:", main_accuracy)

    secondary_accuracy = accuracy_score(y_test_secondary, secondary_predictions)
    print("Secondary Class Accuracy:", secondary_accuracy)

    # Calculate and print the overall accuracy
    overall_accuracy = (main_accuracy + secondary_accuracy) / 2
    print("Overall Accuracy:", overall_accuracy)
    

    # Visualize a few test samples and their predicted classes
    for i in range(2):
        plt.imshow(X_test[i].reshape((2, 6)), cmap='gray')
        plt.title(f"True: {y_test_main[i]}, Pred: {main_predictions[i]}")
        plt.show()

def main():
    # Set image dimensions (2x6 images with 1 channel)
    image_size = (2, 6, 1)
    test_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv'
    save_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\UMVD-FSL\\'
    evaluate_umvd_fsl(test_path, image_size, save_path)

if __name__ == "__main__":
    main()
