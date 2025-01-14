import pandas as pd

# Load the dataset
train_data = pd.read_csv(r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv')
test_data = pd.read_csv(r'C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f.csv')

# Function to save 1 sample from each class (main_class) in the dataset
def save_samples(data, file_path):
    samples = data.groupby('main_class').apply(lambda x: x.sample(min(len(x), 1))).reset_index(drop=True)
    samples.to_csv(file_path, index=False)

# Save 1 sample from each class in train data
save_samples(train_data, r'C:\Users\User\Attack-Detection-Project\datasets\MTA\train_small.csv')

