# import tensorflow as tf
# from tensorflow.keras import layers, Model
# import numpy as np
# import pandas as pd
# import joblib  
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# # Define base CNN
# def create_base_cnn(input_shape):
#     input = layers.Input(shape=input_shape)
#     x = layers.Conv1D(64, kernel_size=3, activation='relu')(input)
#     x = layers.MaxPooling1D(pool_size=2)(x)
#     x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     return Model(input, x)

# # Build Siamese Network
# def build_siamese_network(input_shape, num_classes):
#     base_cnn = create_base_cnn(input_shape)

#     input_a = layers.Input(shape=input_shape)
#     input_b = layers.Input(shape=input_shape)

#     encoded_a = base_cnn(input_a)
#     encoded_b = base_cnn(input_b)

#     distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
#     output = layers.Dense(num_classes, activation='softmax')(distance)

#     return Model([input_a, input_b], output)

# # Train Siamese Network
# def train_and_save_siamese_network(train_path, save_path):
#     # Load and preprocess dataset
#     train_data = pd.read_csv(train_path)
#     X_train = train_data.iloc[:, :-2].values
#     y_train_main = train_data[['main_class']].values
#     y_train_secondary = train_data[['second_class']].values

#     le_main = LabelEncoder()
#     le_secondary = LabelEncoder()
#     y_train_main = le_main.fit_transform(y_train_main)
#     y_train_secondary = le_secondary.fit_transform(y_train_secondary)


#     # Ensure labels are integers
#     y_train_main = y_train_main.astype(int)
#     y_train_secondary = y_train_secondary.astype(int)

#     # Scale features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)

#     # Save encoders and scaler
#     joblib.dump(le_main, save_path + 'le_main.pkl')
#     joblib.dump(le_secondary, save_path + 'le_secondary.pkl')
#     joblib.dump(scaler, save_path + 'scaler.pkl')

#     # Build and compile Siamese Network
#     siamese_network_main = build_siamese_network((X_train.shape[1], 1), len(le_main.classes_))
#     siamese_network_main.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     siamese_network_secondary = build_siamese_network((X_train.shape[1], 1), len(le_secondary.classes_))
#     siamese_network_secondary.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # Train Siamese Network
#     siamese_network_main.fit([X_train, X_train], y_train_main, epochs=50)
#     siamese_network_secondary.fit([X_train, X_train], y_train_secondary, epochs=50)

#     # Save Siamese Network
#     siamese_network_main.save(save_path + 'siamese_network_main.keras')
#     siamese_network_secondary.save(save_path + 'siamese_network_secondary.keras')






    
# def main():
#     train_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv'
#     save_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\\'
#     train_and_save_siamese_network(train_path, save_path)



# if __name__ == "__main__":
#     main()


import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Define base CNN
def create_base_cnn(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation='relu')(input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return Model(input, x)

# Build Siamese Network
def build_siamese_network(input_shape, num_classes):
    base_cnn = create_base_cnn(input_shape)

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    encoded_a = base_cnn(input_a)
    encoded_b = base_cnn(input_b)

    # Use custom distance layer instead of Lambda
    distance = L1Distance()([encoded_a, encoded_b])
    output = layers.Dense(num_classes, activation='softmax')(distance)

    return Model([input_a, input_b], output)

# Train Siamese Network
def train_and_save_siamese_network(train_path, save_path):
    # Load and preprocess dataset
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, :-2].values
    y_train_main = train_data[['main_class']].values
    y_train_secondary = train_data[['second_class']].values

    le_main = LabelEncoder()
    le_secondary = LabelEncoder()
    y_train_main = le_main.fit_transform(y_train_main)
    y_train_secondary = le_secondary.fit_transform(y_train_secondary)

    # Ensure labels are integers
    y_train_main = y_train_main.astype(int)
    y_train_secondary = y_train_secondary.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Reshape for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Save encoders and scaler
    joblib.dump(le_main, save_path + 'le_main.pkl')
    joblib.dump(le_secondary, save_path + 'le_secondary.pkl')
    joblib.dump(scaler, save_path + 'scaler.pkl')

    # Register the custom layer
    tf.keras.utils.get_custom_objects()['L1Distance'] = L1Distance

    # Build and compile Siamese Network
    siamese_network_main = build_siamese_network((X_train.shape[1], 1), len(le_main.classes_))
    siamese_network_main.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    siamese_network_secondary = build_siamese_network((X_train.shape[1], 1), len(le_secondary.classes_))
    siamese_network_secondary.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train Siamese Network
    siamese_network_main.fit([X_train, X_train], y_train_main, epochs=50)
    siamese_network_secondary.fit([X_train, X_train], y_train_secondary, epochs=50)

    # Save Siamese Network
    siamese_network_main.save(save_path + 'siamese_network_main.keras')
    siamese_network_secondary.save(save_path + 'siamese_network_secondary.keras')

def main():
    train_path = r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv'
    save_path = r'C:\Users\User\Attack-Detection-Project\models\saved_models\COD_SNN\\'
    train_and_save_siamese_network(train_path, save_path)

if __name__ == "__main__":
    main()