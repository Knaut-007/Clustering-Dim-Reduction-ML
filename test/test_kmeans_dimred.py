from main import selected_features, n_clusters, df_scaled

def test_kmeans_dimred():
    assert selected_features.shape[1] == n_clusters # Number of selected features should match n_clusters
    assert selected_features.shape[0] == df_scaled.shape[0] # Number of samples should remain the same
    print("K-Means dimensionality reduction test passed.")

if __name__ == "__main__":
    test_kmeans_dimred()
