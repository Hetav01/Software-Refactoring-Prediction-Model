import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
# from configs import BALANCE_DATASET_STRATEGY



def perform_balancing(x, y, strategy=None):
    """
    Performs under/over sampling, according to the number of true and false instances of the x, y dataset.
    :param x: feature values
    :param y: labels
    :return: a balanced x, y
    """

    if strategy is None:
        raise Exception("still don't have a strategy?")         #set at random right now. BALANCE_DATASET_STRATEGY

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
    new_y = pd.DataFrame(new_y, columns=[y.name])
    return new_x, new_y


