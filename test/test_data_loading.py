import sys
sys.path.append("..")  # To import from parent directory

from main import load_data

def test_load_data():
    df, y = load_data()
    assert df.shape[1] == 561 # Feature count should be 561
    assert len(df) == len(y) # Number of samples and labels should match
    print("Data loading test passed.")

if __name__ == "__main__":
    test_load_data()
