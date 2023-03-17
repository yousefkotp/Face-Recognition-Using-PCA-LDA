# Face Recognition
A face recognition project using PCA and LDA algorithms.

> This readME file is a summary of the project. For more details, please refer to the [notebook](faces_vs_nonfaces.ipynb).
## Table of Contents
- [Face Recognition](#face-recognition)
  * [Dataset](#dataset)
    + [Data Splitting](#data-splitting)
  * [Algorithms](#algorithms)
    + [PCA](#pca)
    + [LDA](#lda)
        - [Pseudo Code](#pseudo-code)
  * [Results](#results)
  * [Contributers](#contributers)

## Dataset
- Our dataset for this project is the [AT&T Face Database](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces). The dataset is open-source and can be downloaded from Kaggle.
- The dataset contains 400 images of 40 people. Each person has 10 images. The images are of size 92x112 pixels. The images are in grayscale. The images are in the form of a numpy array. The images are stored in the `archive` folder.

### Data Splitting
- We tried splitting the dataset into training and testing sets. one split was 50-50 and the other one is 70-30. The results are discussed inside the [notebook](faces_vs_nonfaces.ipynb) and in the [Results](#results) section.

## Algorithms 
- Two algorithms were used in the facial recognition for the mentioned dataset which are:
1- PCA: Principal Component Analysis
2- LDA: Linear Discriminant Analysis

### PCA

### LDA 
- Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset while maintaining the class separability. LDA is a supervised technique, meaning that it uses the class labels to perform the dimensionality reduction. LDA is a popular technique for dimensionality reduction in the field of pattern recognition and machine learning. 
#### Pseudo Code
- The pseudo code for the multi-class LDA is as follows:
```python
    # Step 1: Compute the overall mean of the training set
    overall_mean = compute_mean(training_set)

    # Step 2: Compute the between-class scatter matrix and the within-class scatter matrix
    S_B = compute_between_class_scatter(training_set, overall_mean)
    S_W = compute_within_class_scatter(training_set)

    # Step 3: Compute the eigenvalues and eigenvectors of the generalized eigenvalue problem
    eigenvalues, eigenvectors = compute_generalized_eigen(S_B, S_W)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

    # Step 5: Take only the dominant eigenvectors
    new_space = select_eigenvectors(sorted_eigenvectors)

    # Step 6: Return the dominant eigenvectors
    return new_space
```
## Results

## Contributers

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohammed Farid](https://github.com/MohamedFarid612)

- [Adham Mohammed](https://github.com/adhammohamed1)

