import numpy as np
import pandas as pd 
import pickle
import json 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path:str)->pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}:{e}")
#test_data=pd.read_csv("./data/processed/test_processed.csv")


def prepare_data(data:pd.DataFrame)->tuple[pd.DataFrame,pd.Series]:
    try:
        x=data.iloc[:,0:-1].values
        y=data.iloc[:,-1].values
        return x,y
    except Exception as e:
        raise Exception(f"Error preparing data:{e}")
#x_test=test_data.iloc[:,0:-1].values
#y_test=test_data.iloc[:,-1].values


def load_model(file_path:str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {file_path}:{e}")
#model=pickle.load(open("model.pkl","rb"))

def evaluation_method(model,x_test:pd.DataFrame,y_test:pd.DataFrame):
    try:
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        metrics_dict = {
            "acc": acc,
            "pre": pre,
            "recall": recall,
            "f1_score": f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error during evaluation:{e}")

"""
y_pred=model.predict(x_test)

acc=accuracy=accuracy_score(y_test,y_pred)
pre=precision=precision_score(y_test,y_pred)
recall=recall=recall_score(y_test,y_pred)
f1score=f1_score(y_test,y_pred)
metrics_dict={
    "acc":acc,
    "pre":pre,
    "recall":recall,
    "f1_score":f1score
}
"""
def save_metrics(metrics_dict:dict,filepath:str)->None:
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}:{e}")
#with open("metrics.json","wb") as file:
#    json.dump(metrics_dict,file,indent=4)"


def main():
    try:
        test_data_path="./data/processed/test_processed.csv"
        model_path="models/model.pkl"
        metrics_path="reports/metrics.json"

        test_data=load_data(test_data_path)
        x_test, y_test = prepare_data(test_data)
        model=load_model(model_path)
        metrics=evaluation_method(model,x_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An error occurred:{e}")
    
if __name__ == "__main__":
    main()