import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import joblib

# Load and preprocess dataset
def load_data_as_images(train_path, image_size, save_path):
    train_data = pd.read_csv(train_path)

    # Features and labels
    X_train = train_data.iloc[:, :-2].values
    y_train_main = train_data[['main_class']].values
    y_train_secondary = train_data[['second_class']].values

    # Normalize features to [0, 1] for image transformation
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Reshape into grayscale images
    X_train_images = X_train.reshape(-1, *image_size)

    # Encode labels
    le_main = LabelEncoder()
    le_secondary = LabelEncoder()
    y_train_main = le_main.fit_transform(y_train_main.ravel())
    y_train_secondary = le_secondary.fit_transform(y_train_secondary.ravel())

    # Ensure labels are integers
    y_train_main = y_train_main.astype(int)
    y_train_secondary = y_train_secondary.astype(int)

    # Save encoders and scaler
    joblib.dump(le_main, save_path + 'le_main.pkl')
    joblib.dump(le_secondary, save_path + 'le_secondary.pkl')
    joblib.dump(scaler, save_path + 'scaler.pkl')

    return X_train_images, y_train_main, y_train_secondary, len(le_main.classes_), len(le_secondary.classes_)

# Build CNN feature extractor
def build_feature_extractor(input_shape):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (1, 2), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (1, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),  # Latent space
        layers.BatchNormalization()
    ])
    return model

# Compute class prototypes
def compute_prototypes(features, labels, num_classes):
    prototypes = []
    for class_id in range(num_classes):
        class_features = features[labels == class_id]
        prototype = np.mean(class_features, axis=0)
        prototypes.append(prototype)
    return np.array(prototypes)

# Visualize prototype features
def visualize_prototype(prototype, title, subplot_idx, num_cols=5):
    plt.subplot(2, num_cols, subplot_idx)
    # Reshape to a more suitable dimension for visualization (e.g., 8x16)
    reshaped_dim = (8, 16)  # 128 = 8 * 16
    plt.imshow(prototype.reshape(reshaped_dim), cmap='viridis', aspect='auto')
    plt.title(title)
    plt.axis('off')

# Train and evaluate UMVD-FSL
def train_umvd_fsl(X_train, y_train_main, y_train_secondary, num_main_classes, num_secondary_classes, image_size, save_path):
    feature_extractor = build_feature_extractor(image_size)
    
    # Extract features for training images
    train_features = feature_extractor.predict(X_train)

    # Compute prototypes for main and secondary classes
    main_prototypes = compute_prototypes(train_features, y_train_main, num_main_classes)
    secondary_prototypes = compute_prototypes(train_features, y_train_secondary, num_secondary_classes)

    # Save the prototypes
    np.save(save_path + 'main_prototypes.npy', main_prototypes)
    np.save(save_path + 'secondary_prototypes.npy', secondary_prototypes)

    # Save the feature extractor model
    feature_extractor.save(save_path + 'feature_extractor.h5')

    # Visualize prototypes
    plt.figure(figsize=(15, 6))
    
    # Determine number of prototypes to show (minimum of 5 or actual number)
    n_show = min(5, min(len(main_prototypes), len(secondary_prototypes)))
    
    # Visualize main class prototypes
    for i in range(n_show):
        visualize_prototype(main_prototypes[i], f"Main {i}", i+1)
    
    # Visualize secondary class prototypes
    for i in range(n_show):
        visualize_prototype(secondary_prototypes[i], f"Secondary {i}", i+n_show+1)
    
    plt.tight_layout()
    plt.show()

def main():
    # Set image dimensions (2x6 images with 1 channel)
    image_size = (2, 6, 1)
    X_train, y_train_main, y_train_secondary, num_main_classes, num_secondary_classes = load_data_as_images(
        r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv',
        image_size,
        r'C:\Users\User\Attack-Detection-Project\models\saved_models\UMVD-FSL\\'
    )
    train_umvd_fsl(X_train, y_train_main, y_train_secondary, num_main_classes, num_secondary_classes, image_size, r'C:\Users\User\Attack-Detection-Project\models\saved_models\UMVD-FSL\\')

if __name__ == "__main__":
    main()