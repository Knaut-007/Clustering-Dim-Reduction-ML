import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
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


# Encode Class Labels 
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)

# Scale Features 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

from sklearn.naive_bayes import GaussianNB

# Split Data 
X_train_full, X_test_full, y_train, y_test = train_test_split(
    df_scaled, encoded_y, test_size=0.2, random_state=42
)

# Baseline Model: Gaussian Naive Bayes with All Features
start_time = time.time()
classifier_pipeline_full = Pipeline([
    ('classifier', GaussianNB())
])
classifier_pipeline_full.fit(X_train_full, y_train)
y_pred_full = classifier_pipeline_full.predict(X_test_full)
end_time = time.time()

full_features_time = end_time - start_time
accuracy_full = accuracy_score(y_test, y_pred_full)

print(f"\nBaseline Model (All Features):")
print(f"Accuracy: {accuracy_full:.4f}")
print(f"Training Time: {full_features_time:.4f} seconds")
print(f"Number of Features: {X_train_full.shape[1]}")

# K-Means Clustering for Dimensionality Reduction

from sklearn.cluster import KMeans

n_clusters = 50  # You can vary this later (e.g., 40, 50, 60)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(df_scaled.T)  # Transpose: features become data points

# Select one representative feature per cluster
selected_features_indices = [
    np.random.choice(np.where(kmeans.labels_ == i)[0])
    for i in range(n_clusters)
]
selected_features = df_scaled[:, selected_features_indices]

# Split Data with Reduced Features 
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    selected_features, encoded_y, test_size=0.2, random_state=42
)

# Train Gaussian Naive Bayes on Reduced Features 
start_time = time.time()
classifier_pipeline_reduced = Pipeline([
    ('classifier', GaussianNB())
])
classifier_pipeline_reduced.fit(X_train_reduced, y_train)
y_pred_reduced = classifier_pipeline_reduced.predict(X_test_reduced)
end_time = time.time()

reduced_features_time = end_time - start_time
accuracy_reduced = accuracy_score(y_test, y_pred_reduced)

print(f"\nModel with Reduced Features (K-Means):")
print(f"Accuracy: {accuracy_reduced:.4f}")
print(f"Training Time: {reduced_features_time:.4f} seconds")
print(f"Number of Features: {n_clusters}")


# Comparing Results and Plotting

import matplotlib.pyplot as plt

# Test different cluster sizes

def cluster_experiment(cluster_sizes, df_scaled, encoded_y):
    accuracies = []
    training_times = []

    for n_clusters in cluster_sizes:
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(df_scaled.T)

        # Select representative features
        selected_features_indices = [np.random.choice(np.where(kmeans.labels_ == i)[0]) for i in range(n_clusters)]
        selected_features = df_scaled[:, selected_features_indices]

        # Split data
        X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(selected_features, encoded_y, test_size=0.2, random_state=42)

        # Train model
        start_time = time.time()
        classifier_pipeline_reduced = Pipeline([('classifier', GaussianNB())])
        classifier_pipeline_reduced.fit(X_train_reduced, y_train)
        y_pred_reduced = classifier_pipeline_reduced.predict(X_test_reduced)
        end_time = time.time()

        # Record accuracy and time
        accuracy = accuracy_score(y_test, y_pred_reduced)
        training_time = end_time - start_time

        accuracies.append(accuracy)
        training_times.append(training_time)
    return accuracies, training_times

if __name__ == "__main__":
    cluster_sizes = [40, 50, 60]
    accuracies, training_times = cluster_experiment(cluster_sizes, df_scaled, encoded_y)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'n_clusters': cluster_sizes,
        'accuracy': accuracies,
        'training_time': training_times
    })
    results_df.to_csv('cluster_experiment_results.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cluster_sizes, accuracies, marker='o')
    plt.title('Accuracy vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(cluster_sizes, training_times, marker='o', color='orange')
    plt.title('Training Time vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
