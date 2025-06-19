# Project Notes: Clustering for Dimensionality Reduction in Machine Learning

## 1. Data Loading 

### Dataset Used
- **Human Activity Recognition Using Smartphones** dataset from UCI Machine Learning Repository.
- Contains 561 features (sensor readings) for different physical activities.

### Data Loading code Explanation

# Load train and test data
X_train = pd.read_csv(
    'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt',
    delim_whitespace=True,
    header=None
)
y_train = pd.read_csv(
    'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt',
    delim_whitespace=True,
    header=None
)
X_test = pd.read_csv(
    'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt',
    delim_whitespace=True,
    header=None
)
y_test = pd.read_csv(
    'human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt',
    delim_whitespace=True,
    header=None
)

# Combine train and test for EDA and clustering
X = pd.concat([X_train, X_test], ignore_index=True)
y = pd.concat([y_train, y_test], ignore_index=True).iloc[:,0]
X.columns = feature_names
return X, y


### Theoretical Understanding

- **Why combine train and test?**  
  For exploratory data analysis (EDA) and feature clustering, we want to analyze the entire dataset together. This helps us understand the full feature space and redundancy.
- **Why use `sep='\s+'`?**  
  The `features.txt` is space-separated, and sometimes the number of spaces varies. Using `\s+` (regex for one or more whitespace) ensures we correctly extract columns.
- **Why assign feature names?**  
  Assigning the correct feature names to the columns makes the data easier to interpret and process in later steps.

---

## **2. Data Loading Test**

- We want to verify that the data loads correctly and has the expected shape and feature names.

### Notes on Pandas Warnings

- **SyntaxWarning: invalid escape sequence '\s'**  
  In Python, backslashes in strings can create escape sequence issues. To avoid this, use raw strings (prefix with `r`), e.g., `sep=r'\s+'`.
- **FutureWarning: 'delim_whitespace' is deprecated**  
  Pandas recommends using `sep=r'\s+'` instead of `delim_whitespace=True` for reading whitespace-separated files. This ensures future compatibility.

### Testing Setup Notes

#### Steps Taken for `test_data_loading` to Work

- **Used Python’s `-m` flag** to run the test as a module from the project root:
python -m test.test_data_loading

This ensures imports are resolved relative to the project root, not the test folder.

- **Added an empty `__init__.py` file** in the `test` folder.
This file tells Python to treat the `test` directory as a package, which is required for module-based imports (such as `python -m test.test_data_loading`).  
Without `__init__.py`, Python cannot recognize the `test` folder as a valid module, and running tests with the `-m` flag may fail.

