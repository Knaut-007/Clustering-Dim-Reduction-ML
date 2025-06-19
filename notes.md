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

## 3. Exploratory Data Analysis (EDA)

### Code and Output

- **Dataset shape:** Shows the number of samples and features (should be [10299, 561]).
- **First rows:** Helps verify data format and feature names.
- **Descriptive statistics:** Gives an overview of value ranges and distributions for each feature.
- **Missing values:** Checks for any NaNs or incomplete data.
- **Class distribution:** Shows the number of samples per activity, indicating if the dataset is balanced.

### Theoretical Notes

- **Why EDA?**  
  EDA helps to understand the structure, quality, and distribution of the data before modeling. It’s crucial for identifying issues like missing values, outliers, or class imbalance.
- **Findings?**  
  - The dataset has 561 features and over 10,000 samples.
  - There are no missing values.
  - The class distribution is fairly balanced across activities.

### What if There Were Missing Values?

If the dataset contained missing values, I would have considered the following approaches:

- **Removal:**  
  If only a small number of rows or columns had missing values, I could remove those rows or columns using `dropna()`. This is only advisable if the data loss is minimal and does not bias the dataset.

- **Imputation:**  
  For more substantial missing data, I would impute (fill in) the missing values. Common strategies include:
    - **Mean/Median Imputation:** Replace missing values with the mean or median of the respective feature.
    - **Mode Imputation:** For categorical features, replace missing values with the most frequent value.
    - **Advanced Methods:** Use algorithms like KNN imputation or regression imputation for more accurate estimates.

- **Indicator Variable:**  
  Sometimes, I might add a binary indicator column to flag which values were missing, so the model can learn if missingness itself is informative.

**In this project, no missing values were found, so no imputation or removal was necessary.**

## 4. Label Encoding and Feature Scaling

### Code Explanation

- **Label Encoding:**  
  Many machine learning models require numeric labels. `LabelEncoder` converts the activity labels (which may be strings or integers) into a contiguous range of integers (0, 1, 2, ...).
- **Feature Scaling:**  
  Sensor features have different units and scales. `StandardScaler` standardizes each feature to have mean 0 and standard deviation 1. This is crucial for clustering and many ML algorithms, as it ensures all features contribute equally.

### Theoretical Notes

- **Why encode labels?**  
  Algorithms like Naive Bayes require numeric target variables.
- **Why scale features?**  
  K-Means clustering and Naive Bayes are sensitive to feature scales. Standardizing prevents features with larger ranges from dominating the model.

## 5. Train-Test Split and Baseline Model

### Code Explanation

- **Train-Test Split:**  
  The dataset is split into training (80%) and testing (20%) sets using `train_test_split`. This allows us to evaluate model performance on unseen data.
- **Baseline Model:**  
  Used a `GaussianNB` (Gaussian Naive Bayes) classifier as a baseline. The model is trained on all 561 features.
- **Pipeline:**  
  The pipeline structure allows easy swapping or adding of preprocessing/modeling steps.

### Theoretical Notes

- **Why split the data?**  
  To evaluate how well the model generalizes to new, unseen samples.
- **Why use a baseline model?**  
  The baseline provides a reference point to compare the effectiveness of dimensionality reduction and feature selection later.

## 6. K-Means Clustering for Dimensionality Reduction

### Code Explanation

- **Transposing the Data:**  
  By transposing the feature matrix, each feature (column) becomes a "data point" for clustering.  
  This allows grouping similar features together based on their values across all samples.
- **K-Means Clustering:**  
  Clusters the features into `n_clusters` groups. Each cluster contains similar features (highly correlated or redundant).
- **Selecting Representative Features:**  
  For each cluster, select one feature (randomly) to represent the group, reducing the total number of features.

### Theoretical Notes

- **Why cluster features?**  
  Many features are likely redundant. Clustering helps keep only the most informative, non-redundant features.
- **Why select one feature per cluster?**  
  This ensures diversity in the reduced feature set and avoids information loss from using only highly similar features.

## 7. Training and Evaluating Model on Reduced Features

### Code Explanation

- **Train-test split:**  
  The reduced feature set is split into training and testing sets, just like before.
- **Gaussian Naive Bayes:**  
  The same classifier is trained and evaluated, but now using only the selected representative features from K-Means clustering.
- **Performance Comparison:**  
  Print the accuracy, training time, and number of features used for the reduced model.

### Theoretical Notes

- **Why repeat the split?**  
  To ensure a fair comparison, the same random split and labels are used.
- **What to expect?**  
  The reduced model should be faster to train and may have similar accuracy if redundant features were successfully removed.
