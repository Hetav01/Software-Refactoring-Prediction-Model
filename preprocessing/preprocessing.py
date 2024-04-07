import sys
import os
sys.path.append(os.getcwd())

from collections import Counter
import pandas as pd
from configs import SCALE_DATASET, TEST, FEATURE_REDUCTION, BALANCE_DATASET, DROP_METRICS, \
    DROP_PROCESS_AND_AUTHORSHIP_METRICS, PROCESS_AND_AUTHORSHIP_METRICS, DROP_FAULTY_PROCESS_AND_AUTHORSHIP_METRICS
from preprocessingHelper import perform_fit_scaling, perform_scaling, perform_feature_reduction, perform_balancing
from sklearn.preprocessing import StandardScaler

def get_labelled_instances(scaler= None, allowed_features= None, is_training_data: bool= True):
    
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
    
    # if it's a test run, reduce the dataset to only a random sample.k
    if TEST:
        refactored_df = refactored_df.sample(frac= 0.2)
        non_refactored_df = non_refactored_df.sample(frac= 0.2)
    
    print(refactored_df.info())
    print(non_refactored_df.info())
    
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
    
    print(refactored_df.info())
    print(non_refactored_df.info())
    
    """ 
    code to check which columns are present in one dataframe and not in the other.
    
    # non_refactored_columns = set(non_refactored_df.columns)
    # refactored_columns = set(refactored_df.columns)
    
    # column_difference = non_refactored_columns - refactored_columns
    # print("Columns present in non-refactored df and not in refactored df:")
    # for column in column_difference:
    #     print(column)
    
    # column_difference2 =  refactored_columns - non_refactored_columns
    # print("Columns present in non-refactored df and not in refactored df:")
    # for column in column_difference2:
    #     print(column)
    
    """
    
    # Merge the two dataframes
    merged_df = pd.concat([refactored_df, non_refactored_df], axis=0)
    # Add median to the null values in the merged dataframe.
    merged_df.fillna(merged_df.median(), inplace=True)
    
    """ 
    # print(merged_df[["classNumberOfDefaultFields", "classNumberOfDefaultMethods"]].value_counts())  
    # dropping these two columns as they are almost always 0.
    """
    
    merged_df = merged_df.drop(DROP_METRICS, axis=1)    
    
    X = merged_df.drop("predictions", axis=1)
    y = merged_df["predictions"]
    
    # Applying SMOTE to balance the dataset and populate it.
    if (is_training_data and BALANCE_DATASET):
        print("instances before balancing: ", Counter(y))
        X, y = perform_balancing(X, y, "random")
        assert X.shape[0] == y.shape[0]
        print("instances after balancing: ", Counter(y))
    
    # Scaling the dataset
    if SCALE_DATASET and scaler is None:
        X, scaler = perform_fit_scaling(X)
    elif SCALE_DATASET and scaler is not None:
        X = perform_scaling(X, scaler)

    # Feature reduction
    if is_training_data and FEATURE_REDUCTION:
        X = perform_feature_reduction(X, y)
    
    print("Final shape of the dataset: ", X.shape)
    print(X.head())

get_labelled_instances(scaler= None, allowed_features= None, is_training_data= True)