import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def training():
    #MODEL_DIR = os.environ.get('MODEL_DIR')  # Using os.environ.get() to access environment variables
    #MODEL_FILE_LR = os.environ.get('MODEL_FILE_LR')  # Using os.environ.get() to access environment variables
    #MODEL_PATH_FILE = os.path.join(MODEL_DIR, MODEL_FILE_LR)  # Correcting path joining using os.path.join()

    df = fetch_california_housing()
    X = df.data
    Y = df.target
    titles = df.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'linear_regression_model.joblib')

if __name__ == "__main__":
    training()

