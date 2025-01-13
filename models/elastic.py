import ast
import csv
import json
import os
import re
import warnings
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from elasticsearch import Elasticsearch
import argparse


ds = "my-pcap-dataset" # index name

def delete_index(index_name):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    es.indices.delete(index=index_name)
    
def create_index(index_name, dims, metric = "L2_norm"):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    mapping = {
        "properties": {
            "my-pcap-vector_1": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "max_inner_product"
            },
            "class": {
                "type": "keyword"
            },
            "second_class": {
                "type": "keyword"
            }
        }
    }
    es.indices.create(index=index_name, body={"mappings": mapping})


def insert_to_elastic(my_index1,id, vector, tag,second_tag):
    warnings.filterwarnings('ignore')
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    res = es.index(
        index=my_index1,
        id=id,
        body=f"{{ \"my-pcap-vector_1\": {vector}, \"class\": \"{tag}\", \"second_class\": \"{second_tag}\" }} "
    )
    return res


    

def insert(dataset_name, num):
    df = pd.read_csv(dataset_name+".csv",low_memory=False)
    lst = df.values.tolist()
    index = 1
    for vector in lst:
        tag1 = vector[num]
        tag2 = vector[num+1]
        insert_to_elastic(ds,index,vector[:num],tag1,tag2)
        index += 1
        if index % 100 == 0:
            print(index)
    print("index= ",index)

def test_search(my_index1,vector,tag,tag2):
    warnings.filterwarnings('ignore')
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    knn_query = {"field": "my-pcap-vector_1",
                #  "query_vector": ast.literal_eval(vector),
                "query_vector": vector,
                 "k": 100,
                 "num_candidates": 100
                 }
    fields_query = ["my-pcap-vector_1"]
    res = es.knn_search(index=my_index1,knn=knn_query, fields=fields_query)
    max_score = res["hits"]["max_score"]
    predicted_label = res["hits"]["hits"][0]["_source"]["class"]
    predicted_label2 = res["hits"]["hits"][0]["_source"]["second_class"]
    if predicted_label == "0":
        print(res)
        exit(0)
    return max_score,predicted_label,predicted_label2,tag,tag2


def test(dataset_name, num):
    rows = 0
    prediction = []
    df = pd.read_csv(dataset_name+".csv",low_memory=False)
    for line in df.values.tolist():
        rows += 1
        prediction.append(test_search(ds,line[:num],line[num],line[num+1]))
        if rows % 100 == 0:
            print(rows)
    return prediction

def normalizePath(path):
    path = path.replace(" ", "")
    return path.replace('\\\\', '\\')

    

def main():

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Elasticsearch KNN")
    parser.add_argument("-n", "--num", dest="num", help="Number of features", required=True)
    
    parser.add_argument("-d", "--dataset", dest="dataset", help="Path to dataset")
    parser.add_argument("-p", "--percentage",type = float, dest="percentage", help="Percentage of dataset to use for training")

    parser.add_argument("-t", "--train", dest="train", help="Path to train dataset")
    parser.add_argument("-T", "--test", dest="test_data", help="Path to test dataset")

    parser.add_argument("-m", "--metric", dest="metric", help="Metric to use for KNN", default="l2")

    # parse arguments
    args = parser.parse_args()

    if not args.num:
        parser.error("Please provide number of features")
    
    if args.metric not in ["l2_norm", "cosine", "max_inner_product"]:
        parser.error("Invalid metric. Please use 'l2_norm' or 'cosine' or 'max_inner_product'")

    # initialize the index based on the number of features
    try:
        delete_index(ds)
    except:
        pass
    create_index(ds, int(args.num), args.metric)

    # initalize the predictions
    predictions = []
    
    if args.dataset and args.percentage:
        args.dataset = normalizePath(args.dataset)

        # crate train-test split
        df = pd.read_csv(args.dataset + ".csv", low_memory=False) 
        train_data, test_data = train_test_split(df, test_size=args.percentage)
        train_data.to_csv(args.dataset+"_train.csv", index=False)
        test_data.to_csv(args.dataset+"_test.csv", index=False)
        insert(args.dataset+"_train", int(args.num))
        predictions = test(args.dataset+"_test", int(args.num))
    elif args.train and args.test_data:
        insert(normalizePath(args.train), int(args.num))
        predictions = test(normalizePath(args.test_data), int(args.num))

    else: 
        parser.error("Please provide dataset and percentage or train and test dataset")
    
    # calculate accuracy
    y_true = []
    y_pred = []
    for pred in predictions:
        y_true.append(pred[3])
        y_pred.append(pred[1])

    main_report = classification_report(y_true, y_pred)
    print("Main Class Classification Report:")
    print(main_report)

    main_accuracy = accuracy_score(y_true, y_pred)


    y_true = []
    y_pred = []
    for pred in predictions:
        y_true.append(pred[4])
        y_pred.append(pred[2])

    secondary_report = classification_report(y_true, y_pred)
    print("Secondary Class Classification Report:")
    print(secondary_report)


    print("Main Class Accuracy:", main_accuracy)
    secondary_accuracy = accuracy_score(y_true, y_pred)
    print("Secondary Class Accuracy:", secondary_accuracy)

    overall_accuracy = (main_accuracy + secondary_accuracy) / 2
    print("Overall Accuracy:", overall_accuracy)


if __name__ == "__main__":
    main()


# example usage:
# python models\elastic.py -n 12 --test C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f --train C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f