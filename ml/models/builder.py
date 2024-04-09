import sys
import os
sys.path.append(os.getcwd())

from configs import MODELS
from ml.models.mlModels import LinearSVMRefactoringModel, RandomForestRefactoringModel, DecisionTreeRefactoringModel, LogisticRegressionRefactoringModel, NonLinearSVMRefactoringModel, GaussianNaiveBayesRefactoringModel, ExtraTreeRefactoringModel

def build_models():
    all_available_models = {
        "svm": LinearSVMRefactoringModel(),
        "random-forest": RandomForestRefactoringModel(),
        "decision-tree": DecisionTreeRefactoringModel(),
        "logistic-regression": LogisticRegressionRefactoringModel(),
        "svm-non-linear": NonLinearSVMRefactoringModel(),
        "naive-bayes": GaussianNaiveBayesRefactoringModel(),
        "extra-trees": ExtraTreeRefactoringModel()
    }

    return [all_available_models[model] for model in MODELS]