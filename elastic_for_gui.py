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
    
def create_index(index_name, dims):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    mapping = {
        "properties": {
            "my-pcap-vector_1": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "l2_norm"
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

def insert(dataset_name, num, progress_callback=None):
    df = pd.read_csv(dataset_name,low_memory=False)
    lst = df.values.tolist()
    total = len(lst)
    index = 1
    for vector in lst:
        tag1 = vector[num]
        tag2 = vector[num+1]
        insert_to_elastic(ds,index,vector[:num],tag1,tag2)
        index += 1
        if index % 100 == 0:
            print(index)
        if progress_callback:
            progress_callback(index / total * 100)
    print("index= ",index)

def test(dataset_name, num, progress_callback=None):
    df = pd.read_csv(dataset_name,low_memory=False)
    lst = df.values.tolist()
    rows = 0
    prediction = []
    total = len(lst)
    for line in lst:
        prediction.append(test_search(ds,line[:num],line[num],line[num+1]))
        rows += 1
        if rows % 100 == 0:
            print(rows)
        if progress_callback:
            progress_callback(rows / total * 100)
    return prediction


