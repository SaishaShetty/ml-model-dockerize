import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def training():
    df = fetch_california_housing()
    X = df.data
    Y = df.target
    titles = df.feature_names

    X_train, X_test,y_train, y_test = train_test_split(X,Y,test_size=0.2)
    #print(len(y_train))
    #print(len(y_test))

    model = LinearRegression()
    model.fit(X_train,y_train)

    joblib.dump(model, 'linear_regression_model.joblib')

if __name__ == "__main__":
    training()