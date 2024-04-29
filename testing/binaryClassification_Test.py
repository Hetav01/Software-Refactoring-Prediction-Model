import sys
import os
import time
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings('ignore')

from configs import DATASETS
from ml.models.builder import build_models
from ml.pipelines.binary import BinaryClassificationPipeline, _build_production_model, _evaluate_model,_evaluate_model_for_single_set, evaluate_on_unseen_data
import pandas as pd
from configs import SEARCH, N_CV_SEARCH, N_ITER_RANDOM_SEARCH, TEST_SPLIT_SIZE, VALIDATION_DATASETS, TEST
from ml.utils.output import format_results_single_run
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, train_test_split
# from pipelines.pipelines import MLPipeline
from preprocessing.preprocessing import get_labelled_instances,preprocess_unseen_data
from ml.utils.output import format_best_parameters


print("ML4Refactoring: Binary classification")

# Get the data
X_columns, X, y ,scal, sel= get_labelled_instances()

X = pd.DataFrame(data= X, columns= X_columns)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

# Run models
models = build_models()

super_models = []

def _run_single_model(model_def, X, y, X_train, X_test, y_train, y_test):
    file1 = open("result.txt","a")
    # start_time = time.time()
    model = model_def.model()
    file1.write("Model Name:")
    file1.write(str(model))
    file1.write("\n")

    # perform the search for the best hyper parameters
    param_dist = model_def.params_to_tune()
    search = None

    # choose which search to apply
    if SEARCH == 'randomized':
        search = RandomizedSearchCV(model, param_dist, n_iter=N_ITER_RANDOM_SEARCH, cv=StratifiedKFold(n_splits=N_CV_SEARCH, shuffle=True), n_jobs=-1)
    elif SEARCH == 'grid':
        search = GridSearchCV(model, param_dist, cv=StratifiedKFold(n_splits=N_CV_SEARCH, shuffle=True), iid=False, n_jobs=-1)

    # Train and test the model
    test_scores = _evaluate_model_for_single_set(search, X_train, X_test, y_train, y_test)
    file1.write("Test Scores:")
    file1.write(str(test_scores))
    
    file1.write("\n")
    file1.write("\n")
    # Run cross validation on whole dataset and safe production ready model
    super_model = _build_production_model(model_def, search.best_params_, X, y)
    
    super_models.append(super_model)
    # end_time = time.time()
    file1.close()
    # return the scores and the best estimator
    print(test_scores)

X_cols,X_unseen, y_unseen = preprocess_unseen_data(scal, sel)

for model in super_models:
    file1 = open("result_unseen.txt","a")
    file1.write("Model Name:")
    file1.write(str(model))
    file1.write("\n")
    
    test_scores = evaluate_on_unseen_data(model, X_unseen, y_unseen)
    print("Results for unseen data:")
    print(test_scores)
    file1.write("Test Scores:")
    file1.write("\n")
    file1.write(str(test_scores))
    
    file1.write("\n")
    file1.write("\n")
    file1.close()

    

