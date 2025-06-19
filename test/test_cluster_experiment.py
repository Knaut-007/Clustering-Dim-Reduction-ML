from main import cluster_experiment, df_scaled, encoded_y

def test_cluster_experiment():
    cluster_sizes = [40, 50, 60]
    accuracies, training_times = cluster_experiment(cluster_sizes, df_scaled, encoded_y)
    assert len(accuracies) == len(cluster_sizes), "Should have accuracy for each cluster size"
    assert len(training_times) == len(cluster_sizes), "Should have training time for each cluster size"
    print("Cluster experiment test passed.")

if __name__ == "__main__":
    test_cluster_experiment()
