import sys
import os
sys.path.append(os.getcwd())

from configs import DATASETS
from ml.models.builder import build_models
from ml.pipelines.binary import BinaryClassificationPipeline, _build_production_model, _evaluate_model,_evaluate_model_for_single_set
import pandas as pd
from configs import SEARCH, N_CV_SEARCH, N_ITER_RANDOM_SEARCH, TEST_SPLIT_SIZE, VALIDATION_DATASETS, TEST
from ml.utils.output import format_results_single_run
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV, train_test_split
# from pipelines.pipelines import MLPipeline
from preprocessing.preprocessing import get_labelled_instances
from ml.utils.output import format_best_parameters

print("ML4Refactoring: Binary classification")

# Get the data
X_columns, X, y= get_labelled_instances()[:3]

X = pd.DataFrame(data= X, columns= X_columns)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

# Run models
models = build_models()


def _run_single_model(model_def, X, y, X_train, X_test, y_train, y_test):
    model = model_def.model()

    # perform the search for the best hyper parameters
    param_dist = model_def.params_to_tune()
    print(param_dist)
    search = None

    # choose which search to apply
    if SEARCH == 'randomized':
        search = RandomizedSearchCV(model, param_dist, n_iter=N_ITER_RANDOM_SEARCH, cv=StratifiedKFold(n_splits=N_CV_SEARCH, shuffle=True), n_jobs=-1)
    elif SEARCH == 'grid':
        search = GridSearchCV(model, param_dist, cv=StratifiedKFold(n_splits=N_CV_SEARCH, shuffle=True), n_jobs=-1)

    # Train and test the model
    test_scores = _evaluate_model_for_single_set(search, X_train, X_test, y_train, y_test)

    # Run cross validation on whole dataset and safe production ready model
    super_model = _build_production_model(model_def, search.best_params_, X, y)

    # return the scores and the best estimator
    print(test_scores, super_model)

_run_single_model(models[3], X, y, X_train, X_test, y_train, y_test)
