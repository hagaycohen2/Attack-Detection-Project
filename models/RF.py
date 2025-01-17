# Authors: Hagay Cohen - 206846180 , Imri Shai - 213023500
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse


def train_and_save(train_path, save_path):


    # Load your dataset
    train_data = pd.read_csv(train_path)


    # Assuming your dataset has columns 'feature1', 'feature2', ..., 'main_class', 'second_class'
    features = train_data.drop(['main_class', 'second_class'], axis=1)
    main_class = train_data['main_class']
    second_class = train_data['second_class']





    # Initialize the Random Forest Classifier
    rf_main = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_second = RandomForestClassifier(n_estimators=100, random_state=42)

    # Split the dataset into training and testing sets, notice that already splitted
    X_train = features
    y_train_main = main_class
    y_train_second = second_class

    # Train the model on the main class
    rf_main.fit(X_train, y_train_main)

    # Train the model on the second class
    rf_second.fit(X_train, y_train_second)

    # Save the models
    joblib.dump(rf_main, save_path + 'rf_main.pkl')
    joblib.dump(rf_second, save_path + 'rf_second.pkl')


def main():
    parser = argparse.ArgumentParser(description="Random Forest Training")
    parser.add_argument('-t', '--train_path', dest='train_path', help='Path to training data', required=True)
    parser.add_argument('-s', '--save_path', dest='save_path', help='Path to save model', required=True)
    args = parser.parse_args()
    
    train_path = args.train_path
    save_path = args.save_path

    train_and_save(train_path, save_path)


if __name__ == '__main__':
    main()
    
# example usage:

# python C:\Users\tnrha\Attack-Detection-Project\models\RF.py -t C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv -s C:\Users\tnrha\Attack-Detection-Project\models\saved_models\RF\

