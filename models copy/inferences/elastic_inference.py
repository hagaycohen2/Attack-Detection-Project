import warnings
from elasticsearch import Elasticsearch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report



ds = "my-pcap-dataset_small" # index name


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
    if max_score == None:
        print("Something went wrong...")
        predicted_label = "ood"
        predicted_label2 = "ood"
        return max_score,predicted_label,predicted_label2,tag,tag2
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


def main():
    num_features = 12
    
    test_path = r'C:\Users\tnrha\Attack-Detection-Project\datasets\MTA\test_mta_data_new_12f'
    
    predictions = test(test_path, num_features)

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

    
    


