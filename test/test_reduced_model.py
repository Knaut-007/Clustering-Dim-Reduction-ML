from main import accuracy_reduced, reduced_features_time, n_clusters, X_train_reduced

def test_reduced_model():
    assert 0 <= accuracy_reduced <= 1 # Reduced model accuracy should be between 0 and 1
    assert X_train_reduced.shape[1] == n_clusters # Reduced model should use n_clusters features
    print("Reduced feature model test passed.")

if __name__ == "__main__":
    test_reduced_model()
