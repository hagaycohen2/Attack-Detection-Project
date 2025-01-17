# Attack Detection Project

This project is designed to detect attacks using machine learning models. The project allows users to choose a dataset, split it into training and testing sets, and train a model using the specified number of features and classes. The trained model is then used to make predictions on the test set.

## Authors

- Hagay Cohen 206846180 - [GitHub Profile](https://github.com/HagayCohen2)
- Imri Shai 213023500 - [GitHub Profile](https://github.com/ImriShai)

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

3. Run the script you want for the desierd model - first run train, then infernce to predict, as follows:

 ```bash
    python {path to python file} --train/--test {path to train/test file, for train or inference of the model} --save {path to save location}
```
NOTICE: some of the scripts has different arguments. In each file there is an example at the bottom of how to run, to make it simple.





## Project Structure
- `models`: Contains the train and iference for each model.
- `datasets`: Contains the data for the project.


### Training

The training process involves creating an Elasticsearch index, inserting the training data into the index, and training the model using the specified number of features and classes. The trained model is saved in Elasticsearch DB.
For other models the same idea, but the model is saved using joblib to the directory entered by the user.

### Inference

The inference process involves using the trained model to make predictions on the test set. The predictions are evaluated using various metrics such as accuracy, recall, precision, and F1 score.

### Elasticsearch

Elasticsearch is used as the backend for storing and querying the data. It provides efficient indexing and searching capabilities, which are essential for handling large datasets and performing KNN searches. However, Elasticsearch does not inherently support model training and inference, so these processes are implemented in the Python scripts.
In order to run the elastic script, ElasticSearch database must be installed and running.

## Notes

- Ensure that Elasticsearch is properly configured and running before starting the project.
- IMPORTANT: In order to run the models that are different then elastic, you need to change the path in the code itself, to match your username on your local machine.


