import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_split_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

filepath = r"data\WineQT_1.csv"
X_train, X_test, y_train, y_test = load_and_split_data(filepath)

print(X_train.head())