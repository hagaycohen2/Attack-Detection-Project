# Attack Detection Project

This project is designed to detect attacks using machine learning models. The project allows users to choose a dataset, split it into training and testing sets, and train a model using the specified number of features and classes. The trained model is then used to make predictions on the test set.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- Elasticsearch python module, and Elasticsearch actual server
- numpy
- joblib
- tensorflow
- argparse

## Installation

1. Install Python 3.x from [python.org](https://www.python.org/).
2. Install the required Python packages using pip:
    ```bash
    pip install pandas scikit-learn elasticsearch numpy joblib tensorflow argparse
    ```

## Running the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Attack-Detection-Project.git
    cd Attack-Detection-Project
    ```

2. Start the Elasticsearch server. Ensure that Elasticsearch is running on `http://localhost:9200` with the appropriate authentication.
    In windows, in the elasticsearch config directory, set `xpack.security.enabled` to false, in the elasticsearch.yaml file.

3. Run the script you want for the desired model, models is for full data, models copy is for one sample data.
    

## Project Structure
- `models`: Contains the train and iference for each model.
- `models copy`: Same as above but for the single sample dataset.
- `datasets/Tournament/feature_extraction.py`: Contains functions for extracting features from the dataset using wavelet transformations.

## Explanation

### Training

The training process involves creating an Elasticsearch index, inserting the training data into the index, and training the model using the specified number of features and classes. The trained model is saved in Elasticsearch DB.
For other models the same idea, but the model is saved using joblib.

### Inference

The inference process involves using the trained model to make predictions on the test set. The predictions are evaluated using various metrics such as accuracy, recall, precision, and F1 score.

### Elasticsearch

Elasticsearch is used as the backend for storing and querying the data. It provides efficient indexing and searching capabilities, which are essential for handling large datasets and performing KNN searches. However, Elasticsearch does not inherently support model training and inference, so these processes are implemented in the Python scripts.
In order to run the elastic script, ElasticSearch database must be installed and running.

## Notes

- Ensure that Elasticsearch is properly configured and running before starting the project.
- The project supports multi-class classification, allowing users to specify the number of classes for prediction.
- IMPORTANT: In order to run the models that are different then elastic, you need to change the path in the code itself, to match your username on your local machine.


