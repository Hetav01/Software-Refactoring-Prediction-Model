import sys
import os
sys.path.append(os.getcwd())

# print(os.getcwd())
from collections import Counter
import pandas as pd
from configs import SCALE_DATASET, TEST, FEATURE_REDUCTION, BALANCE_DATASET, DROP_METRICS, \
    DROP_PROCESS_AND_AUTHORSHIP_METRICS, PROCESS_AND_AUTHORSHIP_METRICS, DROP_FAULTY_PROCESS_AND_AUTHORSHIP_METRICS
from preprocessing.preprocessingHelper import perform_fit_scaling, perform_scaling, perform_feature_reduction, perform_balancing
from sklearn.preprocessing import StandardScaler

def get_external_dataset():
    file_path_for_refactored = input("Enter the path of refactored data file: ")
    file_path_for_non_refactored = input("Enter the path of non-refactored data file: ")
    
    try:
        refactored_df = pd.read_csv(file_path_for_refactored)
        non_refactored_df = pd.read_csv(file_path_for_non_refactored)
    except FileNotFoundError:
        print("File not found. Please provide correct file paths.")
        return

    return refactored_df, non_refactored_df

def get_labelled_instances(dataset= None, scaler= None, allowed_features= None, is_training_data: bool= True):
    
    refactored_df = pd.read_csv("dataset/yes_1.csv")
    non_refactored_df = pd.read_csv("dataset/no_1.csv")    

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
    
    # print(refactored_df.info())
    # print(non_refactored_df.info())
    
    # drop the columns with all NaN values in them.
    refactored_df.dropna(axis=1, how='all', inplace=True)
    non_refactored_df.dropna(axis=1, how='all', inplace=True)
    
    print("nulls:--------------------------------")
    print(Counter(refactored_df.isnull().sum()))
    print(Counter(non_refactored_df.isnull().sum()))
    
    # dropping all the columns which are not numeric.
    refactored_df = refactored_df.select_dtypes(include=['number'])
    non_refactored_df = non_refactored_df.select_dtypes(include=['number'])
    
    # Replace null values with median
    refactored_df.fillna(refactored_df.median(), inplace=True)
    non_refactored_df.fillna(non_refactored_df.median(), inplace=True)
    
    # print(refactored_df.info())
    # print(non_refactored_df.info())
    
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
    
    # # Applying SMOTE to balance the dataset and populate it.
    # if (is_training_data and BALANCE_DATASET):
    #     print("instances before balancing: ", Counter(y))
    #     X, y = perform_balancing(X, y, "random")
    #     assert X.shape[0] == y.shape[0]
    #     print("instances after balancing: ", Counter(y))
    
    # Scaling the dataset
    if SCALE_DATASET and scaler is None:
        X, scaler = perform_fit_scaling(X)
    elif SCALE_DATASET and scaler is not None:
        X = perform_scaling(X, scaler)

    # Feature reduction
    if is_training_data and FEATURE_REDUCTION and allowed_features is None:
        X, selector = perform_feature_reduction(X, y)
    elif allowed_features is not None:
        drop_list = [c for c in X.columns.values if c not in allowed_features]
        X = X.drop(drop_list, axis=1)

    
    return X.columns.values, X, y, scaler, selector

def preprocess_unseen_data(scaler, selector):

    unseen_refactored = pd.read_csv("dataset/yes_2.csv")
    unseen_non_refactored = pd.read_csv("dataset/no_2.csv")

    unseen_refactored["predictions"] = 1
    unseen_non_refactored["predictions"] = 0
    
    # drop the columns with all NaN values in them.
    unseen_refactored.dropna(axis=1, how='all', inplace=True)
    unseen_non_refactored.dropna(axis=1, how='all', inplace=True)
    
    print("nulls:--------------------------------")
    print(Counter(unseen_refactored.isnull().sum()))
    print(Counter(unseen_non_refactored.isnull().sum()))
    
    # dropping all the columns which are not numeric.
    unseen_refactored = unseen_refactored.select_dtypes(include=['number'])
    unseen_non_refactored = unseen_non_refactored.select_dtypes(include=['number'])
    
    # Replace null values with median
    unseen_refactored.fillna(unseen_refactored.median(), inplace=True)
    unseen_non_refactored.fillna(unseen_non_refactored.median(), inplace=True)
    
     # Merge the two dataframes
    unseen_data = pd.concat([unseen_refactored, unseen_non_refactored], axis=0)
    
    # Add median to the null values in the merged dataframe.
    unseen_data.fillna(unseen_data.median(), inplace=True)
    
    unseen_data = unseen_data.drop(DROP_METRICS, axis=1)    
    
    X_unseen = unseen_data.drop("predictions", axis=1)
    y_unseen = unseen_data["predictions"]
    
    # Scale the dataset.
    X_unseen = scaler.transform(X_unseen)
    
    # Drop columns not present in the selector.
    selected_columns = X_unseen.columns[selector.get_support()]
    X_unseen = X_unseen[selected_columns]

    
    return X_unseen.columns.values, X_unseen, y_unseen

# get_labelled_instances()