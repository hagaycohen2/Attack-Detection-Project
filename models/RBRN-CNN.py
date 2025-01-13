import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
import joblib

# Load and preprocess the dataset
def load_data(train_path, save_path):
    train_data = pd.read_csv(train_path)

    X_train = train_data.iloc[:, :-2].values  # Features
    y_train_main = train_data[['main_class']].values  # Only use main_class for RBRN
    y_train_secondary = train_data[['second_class']].values
   

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Encode labels
    le_main = LabelEncoder()
    y_train_main = le_main.fit_transform(y_train_main.ravel())

    le_secondary = LabelEncoder()
    y_train_secondary = le_secondary.fit_transform(y_train_secondary.ravel())

    # Ensure labels are integers
    y_train_main = y_train_main.astype(int)
    y_train_secondary = y_train_secondary.astype(int)

    # Save encoders and scaler
    joblib.dump(le_main, save_path + 'le_main.pkl')
    joblib.dump(le_secondary, save_path + 'le_secondary.pkl')
    joblib.dump(scaler, save_path + 'scaler.pkl')

    return X_train, y_train_main, y_train_secondary, len(le_main.classes_), len(le_secondary.classes_)


# Build the CNN-based feature extractor
def build_feature_extractor(input_dim):
    inputs = Input(shape=(input_dim,))  
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation=None)(x)  # Latent space
    return Model(inputs, x, name="FeatureExtractor")

# Build the Relation Network
def build_relation_network(feature_dim, num_classes):
    input_a = Input(shape=(feature_dim,))
    input_b = Input(shape=(feature_dim,))

    # Concatenate features
    combined = layers.Concatenate()([input_a, input_b])

    # Relation layers
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class classification

    return Model([input_a, input_b], output, name="RelationNetwork")

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

# Train the RBRN model
def train_rbrn(X_train, y_train_main, y_train_secondary, save_path, num_main_classes, num_secondary_classes):
    input_dim = X_train.shape[1]

    # Build models
    feature_extractor = build_feature_extractor(input_dim)
    relation_network_main = build_relation_network(feature_dim=32, num_classes=num_main_classes)
    relation_network_secondary = build_relation_network(feature_dim=32, num_classes=num_secondary_classes)

    # Extract features
    train_features = feature_extractor.predict(X_train)

    # Create pairs
    train_pairs, train_pair_labels = create_pairs(train_features, y_train_main)
    train_pairs_secondary, train_pair_labels_secondary = create_pairs(train_features, y_train_secondary)

    # One-hot encode labels
    train_pair_labels = np.eye(num_main_classes)[train_pair_labels]
    train_pair_labels_secondary = np.eye(num_secondary_classes)[train_pair_labels_secondary]

    # Train relation network
    relation_network_main.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    relation_network_secondary.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    relation_network_main.fit([train_pairs[:, 0], train_pairs[:, 1]], train_pair_labels, epochs=10)
    relation_network_secondary.fit([train_pairs_secondary[:, 0], train_pairs_secondary[:, 1]], train_pair_labels_secondary, epochs=10)

    # Save models
    feature_extractor.save(save_path + 'feature_extractor.keras')
    relation_network_main.save(save_path + 'relation_network_main.keras')
    relation_network_secondary.save(save_path + 'relation_network_secondary.keras')


def main():
    train_path = r'C:\Users\user\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv'
    save_path = r'C:\Users\user\Attack-Detection-Project\models\saved_models\RBRN\\'

    # Load and preprocess the dataset
    X_train, y_train_main, y_train_secondary, num_main_classes, num_secondary_classes = load_data(train_path, save_path)

    # Train the RBRN model
    train_rbrn(X_train, y_train_main, y_train_secondary, save_path, num_main_classes, num_secondary_classes)


if __name__ == '__main__':
    main()