import pandas as pd
# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt
from tensorflow import keras


def read_data():
    # Reading the first dataset from a CSV file named "leakage_dataset_train_100.csv" and storing it in a pandas dataframe named df_100
    df_100 = pd.read_csv("Data/leakage_dataset_train_100.csv", header=0)
    # Reading the second dataset from a CSV file named "leakage_dataset_train_1000.csv" and storing it in a pandas dataframe named df_1000
    df_1000 = pd.read_csv("Data/leakage_dataset_train_1000.csv", header=0)

    # Reading the validation dataset from a CSV file named "leakage_dataset_validation_1000.csv" and storing it in a pandas dataframe named df_val
    df_val = pd.read_csv("Data/leakage_dataset_validation_1000.csv", header=0)
    return df_100, df_1000, df_val



def data_from_to_numpy(df_100, df_1000, df_val):
    # List the columns for the features and label for each dataframe
    feature_cols = ["mfc1", "mfc2", "mfc3", "mfc4"]
    label_cols = ["y1", "y2"]

    # Load the 100-row training set and convert the features and label to numpy arrays
    train_100 = df_100[feature_cols + label_cols].to_numpy()
    X_train_100, y_train_100 = train_100[:, :4], train_100[:, 4:]

    # Load the 1000-row training set and convert the features and label to numpy arrays
    train_1000 = df_1000[feature_cols + label_cols].to_numpy()
    X_train_1000, y_train_1000 = train_1000[:, :4], train_1000[:, 4:]

    # Load the validation set and convert the features and label to numpy arrays
    val = df_val[feature_cols + label_cols].to_numpy()
    X_val, y_val = val[:, :4], val[:, 4:]
    return X_train_100, y_train_100, X_train_1000, y_train_1000, X_val, y_val

def std_normalise_data(X_train_100, X_train_1000, X_val):
    # Calculate the standard deviation of each feature for the sample training set
    x_train_100_std = np.std(X_train_100[:, :4], axis=0)
    x_train_1000_std = np.std(X_train_1000[:, :4], axis=0)

    # Calculate the mean of the first three features' standard deviations for the sample training set
    x_train_100_mean = np.mean(x_train_100_std[:3])
    x_train_1000_mean = np.mean(x_train_1000_std[:3])

    # Adjust the fourth feature of the 100 sample training set by subtracting the absolute value of the difference between its standard deviation and the mean of the first three features' standard deviations
    X_train_100[:, 3] -= np.abs(x_train_100_mean - x_train_100_std[3])
    X_train_1000[:, 3] -= np.abs(x_train_1000_mean - x_train_1000_std[3])

    # Normalize the sample training set such that the sum of all values in each row is equal to 1
    X_train_100_norm = X_train_100 / np.sum(X_train_100, axis=1, keepdims=True)
    X_train_1000_norm = X_train_1000 / np.sum(X_train_1000, axis=1, keepdims=True)

    # Normalize the validation set such that the sum of all values in each row is equal to 1
    X_val_norm = X_val / np.sum(X_val, axis=1, keepdims=True)
    return (
        X_train_100_norm,
        X_train_1000_norm,
        X_val_norm,
    )
