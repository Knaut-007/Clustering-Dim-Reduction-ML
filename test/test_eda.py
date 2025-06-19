from main import df, y

def test_eda():
    assert df.shape[1] == 561 # Expected 561 features
    assert df.isnull().sum().sum() == 0 # There should be no missing values
    assert y.nunique() >= 2 # There should be at least 2 classes
    print("EDA test passed.")

if __name__ == "__main__":
    test_eda()
