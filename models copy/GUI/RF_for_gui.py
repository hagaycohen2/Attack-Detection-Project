import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.num_features = 12
        self.model1 = RandomForestClassifier()
        self.model2 = RandomForestClassifier()

    
        
    def train(self, callback=None, target_columns=['main_class', 'second_class']):
        data = pd.read_csv(self.train_path)
        X = data.drop(target_columns, axis=1)
        y = data[target_columns]
        self.model1.fit(X, y['main_class'])
        self.model2.fit(X, y['second_class'])
        if callback:
            callback()
        

    def test(self,test_path = "", callback = None, target_columns=['main_class', 'second_class']):
        data = pd.read_csv(self.test_path)
        X = data.drop(target_columns, axis=1)
        y = data[target_columns]
        y_pred1 = self.model1.predict(X)
        y_pred2 = self.model2.predict(X)
        return [y_pred1, y_pred2, y['main_class'], y['second_class']]

    def predict(self, data):
        return self.model.predict(data)
    
# Example usage:
# rf_model = RandomForestModel('path/to/your/train_data.csv', 'path/to/your/test_data.csv')
# rf_model.train()
# accuracy = rf_model.test()
# print(f'Model accuracy: {accuracy}')