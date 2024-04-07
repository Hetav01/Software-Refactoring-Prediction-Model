import sys
import os
sys.path.append(os.getcwd())

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from ml.models.base import SupervisedMLRefactoringModel
from random import uniform



class DecisionTreeRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"max_depth": [3, 6, 12, 24, None],
                  "max_features": ["auto", "sqrt", "log2", None],
                  "min_samples_split": [2, 3, 5, 10, 11],
                  "splitter": ["best", "random"],
                  "criterion": ["gini", "entropy"]}

    def model(self, best_params=None):
        if best_params is not None:
            return DecisionTreeClassifier(random_state=42, max_depth=best_params["max_depth"],
                                   max_features=best_params["max_features"],
                                   min_samples_split=best_params["min_samples_split"],
                                   splitter=best_params["splitter"],
                                   criterion=best_params["criterion"])

        return DecisionTreeClassifier(random_state=42)


class ExtraTreeRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"max_depth": [3, 6, 12, 24, None],
                  "max_leaf_nodes": [2, 3, 5, 6, 10, None],
                  "max_features": ["auto", "sqrt", "log2", None],
                  "min_samples_split": [2, 3, 4, 5, 10],
                  "min_samples_leaf": [1, 2, 3, 4, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": [10, 50, 100, 150, 200]}

    def model(self, best_params=None):
        if best_params is not None:
            return ExtraTreesClassifier(random_state=42,
                                        max_depth=best_params["max_depth"],
                                        max_leaf_nodes=best_params["max_leaf_nodes"],
                                        max_features=best_params["max_features"],
                                        min_samples_split=best_params["min_samples_split"],
                                        min_samples_leaf=best_params["min_samples_leaf"],
                                        bootstrap=best_params["bootstrap"],
                                        criterion=best_params["criterion"],
                                        n_estimators=best_params["n_estimators"])

        return ExtraTreesClassifier(random_state=42)

class LogisticRegressionRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {
            "max_iter": [50, 100, 200, 300, 500, 1000, 2000, 5000],
            "C": [uniform(0.01, 100) for i in range(0, 10)]}

    def model(self, best_params=None):
        if best_params is not None:
            return LogisticRegression(solver='lbfgs', max_iter=best_params["max_iter"], C=best_params["C"])

        return LogisticRegression(solver='lbfgs')

class GaussianNaiveBayesRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"var_smoothing": [1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]}

    def model(self, best_params=None):
        if best_params is not None:
            return GaussianNB(var_smoothing=best_params["var_smoothing"])

        return GaussianNB()
    
class RandomForestRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"max_depth": [3, 6, 12, 24, None],
                  "max_features": ["auto", "sqrt", "log2", None],
                  "min_samples_split": [2, 3, 4, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": [10, 50, 100, 150, 200]}

    def model(self, best_params=None):
        if best_params is not None:
            return RandomForestClassifier(random_state=42, n_jobs=-1,
                                          max_depth=best_params["max_depth"],
                                          max_features=best_params["max_features"],
                                          min_samples_split=best_params["min_samples_split"],
                                          bootstrap=best_params["bootstrap"],
                                          criterion=best_params["criterion"],
                                          n_estimators=best_params["n_estimators"])

        return RandomForestClassifier(random_state=42)

class LinearSVMRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"C": [uniform(0.01, 10) for i in range(0, 10)],
                  "kernel": ["linear"],
                  "shrinking": [False]
                  }

    def model(self, best_params=None):
        if best_params is not None:
            return SVC(shrinking=False, C=best_params["C"], kernel=best_params["kernel"])

        return SVC(shrinking=False)



class NonLinearSVMRefactoringModel(SupervisedMLRefactoringModel):
    def params_to_tune(self):
        return {"C": [uniform(0.01, 10) for i in range(0, 10)],
                  "kernel": ["poly", "rbf", "sigmoid"],
                  "degree": [2, 3, 5, 7, 10],
                  "gamma": [uniform(0.01, 10) for i in range(0, 10)],
                  "decision_function_shape": ["ovo", "ovr"]}

    def model(self, best_params=None):
        if best_params is not None:
            return SVC(C=best_params["C"], kernel=best_params["kernel"],
                       degree=best_params["degree"], gamma=best_params["gamma"],
                       decision_function_shape=best_params["decision_function_shape"])

        return SVC(shrinking=False)
