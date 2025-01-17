# Authors: Hagay Cohen - 206846180 , Imri Shai - 213023500
import warnings
import pandas as pd
from elasticsearch import Elasticsearch
import argparse

ds = "my-pcap-dataset"

def delete_index(index_name):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    es.indices.delete(index=index_name)
    
def create_index(index_name, dims, metric = "l2_norm"):
    es = Elasticsearch(['http://localhost:9200'], http_auth=('elasticsearch', '3M44Xn9lRLm0ciIVlQ2X1w'), verify_certs=False)
    mapping = {
        "properties": {
            "my-pcap-vector_1": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": metric
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
    df = pd.read_csv(dataset_name,low_memory=False)
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





    

def main():
    global ds
    
    
    num_features = 12

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Elasticsearch KNN")
    parser.add_argument("-m", "--metric", dest="metric", help="Metric to use for KNN", default="l2_norm")
    parser.add_argument('-t', '--train_path', dest='train_path', help='Path to training data', required=True)
    parser.add_argument('-i', '--index', dest='index', help='Index name', default="my-pcap-dataset")
    args = parser.parse_args()
    
    train_path = args.train_path
    ds = args.index


    if args.metric not in ["l2_norm", "cosine", "max_inner_product"]:
        parser.error("Invalid metric. Please use 'l2_norm' or 'cosine' or 'max_inner_product'")

    
    try:
        delete_index(ds) # delete index if it already exists
    except:
        pass
    create_index(ds, num_features, args.metric) # create index with 12 dimensions and desired metric
    
    insert(train_path, num_features)



if __name__ == "__main__":
    main()


# example usage:
# python C:\Users\tnrha\Attack-Detection-Project\models\elastic.py -t C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f.csv -i my-pcap-dataset