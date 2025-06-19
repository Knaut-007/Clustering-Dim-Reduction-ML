from main import accuracy_full, X_train_full

def test_baseline_model():
    assert 0 <= accuracy_full <= 1 # Accuracy should be between 0 and 1
    assert X_train_full.shape[1] == 561 # Baseline model should use all features
    print("Baseline model test passed.")

if __name__ == "__main__":
    test_baseline_model()
