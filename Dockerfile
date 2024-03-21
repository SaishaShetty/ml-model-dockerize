FROM jupyter/scipy-notebook

WORKDIR /my_data

RUN pip install joblib

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python train.py


