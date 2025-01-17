import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers 
import joblib
import argparse

# Load and preprocess dataset
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-2].values  # Features
    y_train = train_data[['main_class', 'second_class']].values  # Labels

    X_test = test_data.iloc[:, :-2].values
    y_test = test_data[['main_class', 'second_class']].values

    # Encode labels
    le_main = LabelEncoder()
    le_secondary = LabelEncoder()
    y_train[:, 0] = le_main.fit_transform(y_train[:, 0])
    y_train[:, 1] = le_secondary.fit_transform(y_train[:, 1])
    y_test[:, 0] = le_main.transform(y_test[:, 0])
    y_test[:, 1] = le_secondary.transform(y_test[:, 1])

    # Ensure labels are integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save encoders and scaler
    joblib.dump(le_main, r'C:\Users\User\Attack-Detection-Project\models copy\saved_models\Hallucinator\le_main.pkl')
    joblib.dump(le_secondary, r'C:\Users\User\Attack-Detection-Project\models copy\saved_models\Hallucinator\le_secondary.pkl')
    joblib.dump(scaler, r'C:\Users\User\Attack-Detection-Project\models copy\saved_models\Hallucinator\scaler.pkl')

    return X_train, y_train, X_test, y_test, len(le_main.classes_), len(le_secondary.classes_)

# Build Hallucinator
def build_hallucinator(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    return model

# Build classifier
def build_classifier(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Train models
def train_models(X_train, y_train, X_test, y_test, main_classes, secondary_classes, save_path):
    input_dim = X_train.shape[1]
    hallucinator = build_hallucinator(input_dim)
    classifier_main = build_classifier(input_dim, main_classes)
    classifier_secondary = build_classifier(input_dim, secondary_classes)

    # Generate synthetic samples
    synthetic_data = hallucinator.predict(X_train)
    combined_data = np.vstack((X_train, synthetic_data))
    combined_main_labels = np.hstack((y_train[:, 0], y_train[:, 0]))
    combined_secondary_labels = np.hstack((y_train[:, 1], y_train[:, 1]))

    # Train classifiers
    classifier_main.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier_secondary.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    classifier_main.fit(combined_data, combined_main_labels, validation_data=(X_test, y_test[:, 0]), epochs=10, batch_size=32)
    classifier_secondary.fit(combined_data, combined_secondary_labels, validation_data=(X_test, y_test[:, 1]), epochs=10, batch_size=32)

    # Save models
    classifier_main.save(save_path + 'Hallucinator_main.keras')
    classifier_secondary.save(save_path + 'Hallucinator_secondary.keras')

def main():
    
    parser = argparse.ArgumentParser(description="Metric Learning - COD_SNN")
    parser.add_argument('-t', '--train_path', dest='train_path', help='Path to training data', required=True)
    parser.add_argument('-s', '--save_path', dest='save_path', help='Path to save model', required=True)
    args = parser.parse_args()
    
    train_path = args.train_path
    save_path = args.save_path
    
    
    X_train, y_train, X_test, y_test, main_classes, secondary_classes = load_data(r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_small.csv', r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv')
    train_models(X_train, y_train, X_test, y_test, main_classes, secondary_classes, r"C:\Users\User\Attack-Detection-Project\models copy\saved_models\Hallucinator\\")

if __name__ == "__main__":
    main()