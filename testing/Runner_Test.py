# Description: This file is used to run a small simulation of the model training process.
import sys
import os
# sys.path.append(os.getcwd())

from preprocessing.preprocessing import get_labelled_instances
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

model = LogisticRegression()

X_columns, X, y= get_labelled_instances()[:3]

X = pd.DataFrame(data= X, columns= X_columns)
# y = pd.DataFrame(data= y, columns= ["predictions"])

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
