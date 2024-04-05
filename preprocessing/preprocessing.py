import sys
import os
sys.path.append(os.getcwd())

from collections import Counter
import pandas as pd
from configs import SCALE_DATASET, TEST, FEATURE_REDUCTION, BALANCE_DATASET, DROP_METRICS, \
    DROP_PROCESS_AND_AUTHORSHIP_METRICS, PROCESS_AND_AUTHORSHIP_METRICS, DROP_FAULTY_PROCESS_AND_AUTHORSHIP_METRICS
from preprocessingHelper import perform_fit_scaling, perform_scaling, perform_feature_reduction

def get_labelled_instances(scaler= None, allowed_features= None, is_training_data: bool= True):
    #print(f"---- Retrive labelled instances for: {df}")
    
    refactored_df = pd.read_csv("/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI644 SWEN for Data Science/Group 7 Project Files/Software-Refactoring-Prediction-Model/dataset/yes_20k.csv")
    non_refactored_df = pd.read_csv("/Users/ajaykumarpatel/Desktop/Data Science/Grad DS Work/DSCI644 SWEN for Data Science/Group 7 Project Files/Software-Refactoring-Prediction-Model/dataset/no_10k.csv")
    
    print(f"---- Refactored dataframe shape: {refactored_df.shape}")
    print(f"---- Non-Refactored dataframe shape: {non_refactored_df.shape}")
    
    # # Drop rows with NaN values.
    # refactored_df = refactored_df.dropna()
    # non_refactored_df = non_refactored_df.dropna()
    
    # if refactored_df.shape[0] == 0 or non_refactored_df.shape[0] == 0:
    #     print("No data available.")
    #     return None, None, None, None       #we will return the x.column.values, x, y, scaler
    
    print(f"---- Refactored dataframe shape after dropping NaN values: {refactored_df.shape}")
    print(f"---- Non-Refactored dataframe shape after dropping NaN values: {non_refactored_df.shape}")

    refactored_df["predictions"] = 1
    non_refactored_df["predictions"] = 0
    
    #merge and shuffle the data
    merged_df = pd.concat([refactored_df, non_refactored_df], axis=0).sample(frac=1, random_state = 42)
    
    print(refactored_df.info())
    print(refactored_df.describe())
    
    print(non_refactored_df.info())
    print(non_refactored_df.describe())
    
    # drop the columns with all NaN values in them.
    refactored_df.dropna(axis=1, how='all', inplace=True)
    
    print("nulls:--------------------------------")
    print(Counter(refactored_df.isnull().sum()))
    print(Counter(non_refactored_df.isnull().sum()))
    
    # dropping all the columns which are not numeric.
    refactored_df = refactored_df.select_dtypes(include=['number'])
    non_refactored_df = non_refactored_df.select_dtypes(include=['number'])
    
    # Replace null values with median
    refactored_df.fillna(refactored_df.median(), inplace=True)
    non_refactored_df.fillna(non_refactored_df.median(), inplace=True)
    
    print(Counter(refactored_df.isnull().sum()))
    print(Counter(non_refactored_df.isnull().sum()))
    
    

get_labelled_instances(scaler= None, allowed_features= None, is_training_data= True)