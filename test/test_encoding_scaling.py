from main import encoded_y, df_scaled, df

def test_encoding_scaling():
    # Check label encoding
    assert encoded_y.min() == 0 # Encoded labels should start at 0
    assert encoded_y.max() == len(set(encoded_y)) - 1 # Encoded labels should be contiguous integers
    # Check scaling (mean ~0, std ~1)
    means = df_scaled.mean(axis=0)
    stds = df_scaled.std(axis=0)
    assert abs(means.mean()) < 1e-2 # Scaled features should have mean close to 0
    assert abs(stds.mean() - 1) < 1e-2 # Scaled features should have std close to 1
    print("Encoding and scaling test passed.")

if __name__ == "__main__":
    test_encoding_scaling()
