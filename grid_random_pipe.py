import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

from sklearn.datasets import make_classification

if __name__== "__main__":
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    scl = preprocessing.StandardScaler
    pca= decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline(
        [
            ("scale", scl),
            ("pca", pca),
            ("rf", rf)
        ]
    )
    param_grid = {
        "pca__n_components": np.arange(5,10),
        "rf__n_estimators": np.arange(100,1500,100),
        "rf__max_depth": np.arange(1,20),
        "rf__criterion": ["gini","entrophy"],

    }

    model = model_selection.RandomizedSearchCV(

        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )

    model.fit(X,Y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())