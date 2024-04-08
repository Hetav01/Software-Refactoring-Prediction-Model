from sklearn.metrics import confusion_matrix

"""
    Plot of a confusion matrix looks like:
        [[TP, FP],
        [FN, TN]]
    
    where: 
        TN: True Negative
        FP: False Positive
        FN: False Negative
        TP: True Positive
    index 0: negative class
    index 1: positive class
    and x axis is actual class and y axis is predicted class.
"""

def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]

def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]

def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]

def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


