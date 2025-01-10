import ast
import csv
import json
import os
import re
import warnings
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from elasticsearch import Elasticsearch
import argparse
import threading


ds = "my-pcap-dataset" # index name

def delete_index(index_name):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    es.indices.delete(index=index_name)
    
def create_index(index_name, dims, num_classes):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    properties = {
        "my-pcap-vector_1": {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "l2_norm"
        }
    }
    for i in range(num_classes):
        properties[f"class_{i+1}"] = {"type": "keyword"}
    mapping = {"properties": properties}
    es.indices.create(index=index_name, body={"mappings": mapping})


def insert_to_elastic(my_index1, id, vector, tags):
    warnings.filterwarnings('ignore')
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    body = {"my-pcap-vector_1": vector}
    for i, tag in enumerate(tags):
        body[f"class_{i+1}"] = tag
    res = es.index(index=my_index1, id=id, body=body)
    return res


def test_search(my_index1, vector, tags):
    warnings.filterwarnings('ignore')
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    knn_query = {
        "field": "my-pcap-vector_1",
        "query_vector": vector,
        "k": 100,
        "num_candidates": 100
    }
    fields_query = ["my-pcap-vector_1"]
    res = es.knn_search(index=my_index1, knn=knn_query, fields=fields_query)
    predicted_labels = [res["hits"]["hits"][0]["_source"][f"class_{i+1}"] for i in range(len(tags))]
    return predicted_labels + tags

def insert(dataset_name, num, num_classes, progress_callback=None):
    df = pd.read_csv(dataset_name, low_memory=False)
    lst = df.values.tolist()
    total = len(lst)
    index = 1
    for vector in lst:
        tags = vector[num:num + num_classes]
        insert_to_elastic(ds, index, vector[:num], tags)
        index += 1
        if index % 100 == 0:
            print(index)
        if progress_callback:
            progress_callback(index / total * 100)
    print("index= ", index)

def test(dataset_name, num, num_classes, progress_callback=None):
    df = pd.read_csv(dataset_name, low_memory=False)
    lst = df.values.tolist()
    rows = 0
    prediction = []
    total = len(lst)
    for line in lst:
        prediction.append(test_search(ds, line[:num], line[num:num + num_classes]))
        rows += 1
        if rows % 100 == 0:
            print(rows)
        if progress_callback:
            progress_callback(rows / total * 100)
    return prediction


