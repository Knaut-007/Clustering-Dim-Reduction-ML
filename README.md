# Clustering for Dimensionality Reduction in Machine Learning

This project demonstrates how to use **K-Means clustering** for dimensionality reduction on high-dimensional sensor data, using the [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset.

## Project Overview

- **Goal:** Reduce the number of features in a dataset with many redundant or irrelevant features, while maintaining or improving classification accuracy and reducing training time.
- **Approach:** 
  - Normalize all features.
  - Transpose the data so that each feature becomes a "data point" for clustering.
  - Use K-Means to cluster similar features.
  - Select one representative feature from each cluster.
  - Train and evaluate a classifier (Gaussian Naive Bayes) on both the full and reduced feature sets.
  - Compare accuracy and training time.

## Workflow

1. **Data Loading & EDA:** Load and explore the dataset, check for missing values, and analyze class distribution.
2. **Preprocessing:** Encode class labels and scale features.
3. **Baseline Model:** Train a Gaussian Naive Bayes classifier on all 561 features.
4. **Dimensionality Reduction:** 
    - Cluster features using K-Means (on the transposed feature matrix).
    - Select a representative feature from each cluster.
    - Train the classifier on the reduced feature set (e.g., 40, 50, 60 features).
5. **Evaluation:** Compare accuracy and training time for baseline and reduced models. Visualize results with plots.

## Key Results

- **Significant reduction in the number of features** (from 561 to as few as 40–60) with minimal or no loss in accuracy.
- **Training time decreased dramatically** for the reduced feature models.
- **Clustering-based feature selection** effectively removes redundant information and preserves the most informative features.

| Model                        | Accuracy | Training Time (s) | Number of Features |
|------------------------------|----------|-------------------|--------------------|
| Baseline (All Features)      | ~0.75    | ~0.12             | 561                |
| Reduced (K-Means, 50 feats.) | ~0.83    | ~0.01             | 50                 |

## Plots

- **Left:** Accuracy vs. Number of Clusters (features)
- **Right:** Training Time vs. Number of Clusters

## Insights

- Many features in high-dimensional sensor data are redundant.
- K-Means clustering can be used for unsupervised feature selection.
- Dimensionality reduction improves computational efficiency and can even improve model accuracy by removing noise.

## Reproducibility

- All code is modular and tested.
- To reproduce results, download the dataset and run `main.py`.
- Plots and results are saved as files in the repository.

## References

- [UCI Machine Learning Repository: Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [Scikit-learn documentation](https://scikit-learn.org/)

---

*Project by Aadarsh Jha.*
