import numpy as np
import pandas as pd
import joblib 
from joblib import dump,load
import os

def perform_inference(input_data):
    dirpath = os.getcwd()
    print("dirpath = ",dirpath)
    output = os.path.join(dirpath, 'output.csv')

    model = joblib.load('linear_regression_model.joblib')  
    prediction = model.predict(input_data)
    new = pd.DataFrame(prediction, columns =["Predictions"])
    new.to_csv(output)
    return prediction

if __name__ == "__main__":
    # Example input for prediction (using features from the California housing dataset)
    exm = np.array([ 8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]).reshape(1, -1)
    prediction = perform_inference(exm)
    print("Prediction:", prediction)

