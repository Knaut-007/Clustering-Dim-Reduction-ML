import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans 
import time

def load_data():
    # Load features
    feature_names = pd.read_csv(
        'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/features.txt',
        sep=r'\s+',
        header=None,
        usecols=[1]
    )[1].tolist()

    # Load train and test data
    X_train = pd.read_csv(
        'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt',
        sep=r'\s+',
        header=None
    )
    y_train = pd.read_csv(
        'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt',
        sep=r'\s+',
        header=None
    )
    X_test = pd.read_csv(
        'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt',
        sep=r'\s+',
        header=None
    )
    y_test = pd.read_csv(
        'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt',
        sep=r'\s+',
        header=None
    )

    # Combine train and test for EDA and clustering
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True).iloc[:,0]
    X.columns = feature_names
    return X, y

df, y = load_data()

# Exploratory Data Analysis (EDA)

# Print dataset shape and first few rows
print("Dataset shape:", df.shape)
print("First 5 rows:")
print(df.head())

# Print basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum().sum())  # Total missing values

# Print class distribution
print("\nClass distribution:")
print(y.value_counts())
