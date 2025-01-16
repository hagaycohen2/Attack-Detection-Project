
import warnings
import pandas as pd
from elasticsearch import Elasticsearch
import argparse


ds = "my-pcap-dataset" # index name

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





    

def main():
    
    
    num_features = 12

    # initialize argument parser
    parser = argparse.ArgumentParser(description="Elasticsearch KNN")
    parser.add_argument("-m", "--metric", dest="metric", help="Metric to use for KNN", default="l2_norm")

    # parse arguments
    args = parser.parse_args()

    
    if args.metric not in ["l2_norm", "cosine", "max_inner_product"]:
        parser.error("Invalid metric. Please use 'l2_norm' or 'cosine' or 'max_inner_product'")

    
    try:
        delete_index(ds) # delete index if it already exists
    except:
        pass
    create_index(ds, num_features, args.metric) # create index with 12 dimensions and desired metric
    
    train_path = r'C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f'

    insert(train_path, num_features)



if __name__ == "__main__":
    main()


# example usage:
# python models\elastic.py -n 12 --test C:\Users\User\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f --train C:\Users\User\Attack-Detection-Project\datasets\MTA\train_mta_data_new_12f