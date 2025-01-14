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

class ElasticSearchModel:
    def __init__(self,test_path ,train_path , es_host='http://localhost:9200', es_user='elasticsearch', es_pass='3M44Xn9lRLm0ciIVlQ2X1w'):
        self.index_name = "MTA"
        self.dims = 12
        self.train_path = train_path
        self.test_path = test_path
        self.es = Elasticsearch([es_host], http_auth=(es_user, es_pass), verify_certs=False)
        try:
            self.delete_index()
        except:
            pass
        self.create_index(self.dims)

    def delete_index(self):
        self.es.indices.delete(index=self.index_name)

    def create_index(self, dims):
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
        self.es.indices.create(index=self.index_name, body={"mappings": mapping})

    def insert_to_elastic(self, id, vector, tag, second_tag):
        warnings.filterwarnings('ignore')
        res = self.es.index(
            index=self.index_name,
            id=id,
            body=f"{{ \"my-pcap-vector_1\": {vector}, \"class\": \"{tag}\", \"second_class\": \"{second_tag}\" }} "
        )
        return res

    def test_search(self, vector, tag, tag2):
        warnings.filterwarnings('ignore')
        knn_query = {"field": "my-pcap-vector_1",
                     "query_vector": vector,
                     "k": 100,
                     "num_candidates": 100
                     }
        fields_query = ["my-pcap-vector_1"]
        res = self.es.knn_search(index=self.index_name, knn=knn_query, fields=fields_query)
        max_score = res["hits"]["max_score"]
        predicted_label = res["hits"]["hits"][0]["_source"]["class"]
        predicted_label2 = res["hits"]["hits"][0]["_source"]["second_class"]
        if predicted_label == "0":
            print(res)
            exit(0)
        return max_score, predicted_label, predicted_label2, tag, tag2

    def train(self, num, progress_callback=None):
        df = pd.read_csv(self.train_path, low_memory=False)
        lst = df.values.tolist()
        total = len(lst)
        index = 1
        for vector in lst:
            tag1 = vector[num]
            tag2 = vector[num + 1]
            self.insert_to_elastic(index, vector[:num], tag1, tag2)
            index += 1
            if index % 100 == 0:
                print(index)
            if progress_callback:
                progress_callback(index / total * 100)
        print("index= ", index)

    def test(self, dataset_name, num, progress_callback=None):
        df = pd.read_csv(dataset_name, low_memory=False)
        lst = df.values.tolist()
        rows = 0
        prediction = []
        total = len(lst)
        for line in lst:
            prediction.append(self.test_search(line[:num], line[num], line[num + 1]))
            rows += 1
            if rows % 100 == 0:
                print(rows)
            if progress_callback:
                progress_callback(rows / total * 100)
        return prediction

# Example usage:
# model = ElasticSearchModel("my-pcap-dataset")
# model.create_index(dims=128)
# model.train("dataset.csv", num=128)
# predictions = model.test("dataset.csv", num=128)


