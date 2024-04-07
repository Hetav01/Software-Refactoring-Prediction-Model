import sys
import os

from sklearn.feature_selection import RFECV
sys.path.append(os.getcwd())

from configs import BALANCE_DATASET_STRATEGY, N_CV_FEATURE_REDUCTION
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
#import configs
#from configs import N_CV_FEATURE_REDUCTION
#from utils.log import log

def perform_fit_scaling(x):
    """
    Scales all the values between [0,1]. It often speeds up the learning process.

    :param x: the feature values
    :return: x, scaled
    """

    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    columns = x.columns
    x = scaler.fit_transform(x)
    x = pd.DataFrame(x, columns=columns)  # keeping the column names

    return x, scaler


def perform_scaling(x, scaler):
    """
    Scales all the values between [0,1]. It often speeds up the learning process.

    :param x: the feature values
    :param scaler: a predefined and fitted scaler, e.g. a MinMaxScaler
    :return: x, scaled
    """
    columns = x.columns
    x = scaler.fit_transform(x)
    x = pd.DataFrame(x, columns=columns)  # keeping the column names

    return x


def perform_feature_reduction(x, y):
    """
    Performs feature reduction in the x, y

    For now, it uses linear SVR as estimator, and removes feature by feature.

    :param x: feature values
    :param y: labels
    :return: x, y, where x only contain the relevant features.
    """

    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv= N_CV_FEATURE_REDUCTION)    #N_CV_FEATURE_REDUCTION

    # log("Features before reduction (total of {}): {}".format(len(x.columns.values), ', '.join(x.columns.values)))
    selector.fit(x, y)
    x = x[x.columns[selector.get_support(indices=True)]] # keeping the column names

    # log("Features after reduction (total of {}): {}".format(len(x.columns.values), ', '.join(x.columns.values)))
    # log("Feature ranking: {}".format(', '.join(str(e) for e in selector.ranking_)))
    # log("Feature grid scores: {}".format(', '.join(str(e) for e in selector.grid_scores_)))

    return x


def perform_balancing(x, y, strategy=None):
    """
    Performs under/over sampling, according to the number of true and false instances of the x, y dataset.
    :param x: feature values
    :param y: labels
    :return: a balanced x, y
    """

    if strategy is None:
        strategy = BALANCE_DATASET_STRATEGY # raise Exception("still don't have a strategy?")         #set at random right now. BALANCE_DATASET_STRATEGY

    if strategy == 'random':
        rus = RandomUnderSampler(random_state=42)
    elif strategy == 'oversampling':
        rus = SMOTE(random_state=42)
    elif strategy == 'cluster_centroids':
        rus = ClusterCentroids(random_state=42)
    elif strategy == 'nearmiss':
        rus = NearMiss(version=1)
    else:
        raise Exception("algorithm not found")

    new_x, new_y = rus.fit_resample(x, y)
    new_x = pd.DataFrame(new_x, columns=x.columns)
    # new_y = pd.DataFrame(new_y, columns=[y.name])
    return new_x, new_y
