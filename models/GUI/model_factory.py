from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import models
import models.elastic_for_gui
import models.RF_for_gui

class ModelFactory:
    def __init__(self, ):
        pass
        

    def get_model(self,model_type, train_path, test_path):
        if model_type == "Random forest":
            return models.RF_for_gui.RandomForestModel(train_path, test_path)
        elif model_type == "Elastic ANN":
            return models.elastic_for_gui(train_path, test_path)
        
        else:
            return None
        
    

