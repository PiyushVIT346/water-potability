import pandas as pd 
import numpy as np
import os

def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")

#train_data=pd.read_csv("./data/raw/train.csv")
#test_data=pd.read_csv("./data/raw/test.csv")
"""
#version1
def fill_missing_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value=df[column].mean()
                df[column].fillna(mean_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values with mean: {e}")
def save_data(df:pd.DataFrame,filepath:str)->None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")
#train_processed_data=fill_missing_with_median(train_data)
#test_processed_data=fill_missing_with_median(test_data)
"""
#version 2
def fill_missing_with_median(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value=df[column].median()
                df[column].fillna(mean_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values with median: {e}")
def save_data(df:pd.DataFrame,filepath:str)->None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

def main():
    try:
        raw_data_path="./data/raw"
        processed_data_path="./data/processed"
        train_data=load_data(os.path.join(raw_data_path,"train.csv"))
        test_data=load_data(os.path.join(raw_data_path,"test.csv"))
        #train_processed_data=fill_missing_with_mean(train_data)
        #train_processed_data=fill_missing_with_mean(train_processed_data)
        #test_processed_data=fill_missing_with_mean(test_data)
        train_processed_data=fill_missing_with_median(train_data)
        train_processed_data=fill_missing_with_median(train_processed_data)
        test_processed_data=fill_missing_with_median(test_data)
        os.makedirs(processed_data_path)
        #version with mean
        #save_data(train_processed_data,os.path.join(processed_data_path,"train_processed.csv"))
        #save_data(test_processed_data,os.path.join(processed_data_path,"test_processed.csv"))

        #version with median
        save_data(train_processed_data,os.path.join(processed_data_path,"train_processed_median.csv"))
        save_data(test_processed_data,os.path.join(processed_data_path,"test_processed_median.csv"))
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
#data_path=os.path.join("data","processed")
#os.makedirs(data_path)
#train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
#test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)

if __name__ == "__main__":
    main()
