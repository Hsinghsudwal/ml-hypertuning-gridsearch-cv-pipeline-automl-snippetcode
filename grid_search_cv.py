import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets

from sklearn.datasets import make_classification

if __name__== "__main__":
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    #X, Y = make_classification(n_samples=200, n_classes=2, n_features=10, n_redundant=0, random_state=1)

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [1,3,5],
        "criterion": ["gini","entrophy","log_loss"],

    }

    model = model_selection.GridSearchCV(

        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    model.fit(X,Y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())