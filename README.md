# Face Recognition
A face recognition project using PCA and LDA algorithms.

> This readme file is a summary of the project. For more details, please refer to the [notebook](faces_vs_nonfaces.ipynb).
## Table of Contents
- [Face Recognition](#face-recognition)
  * [Dataset](#dataset)
    + [Data Splitting](#data-splitting)
  * [Algorithms](#algorithms)
    + [PCA](#pca)
    + [LDA](#lda)
        - [Pseudo Code](#pseudo-code)
        - [Using K-NN Classifier after LDA](#using-k-nn-classifier-after-lda)
  * [Comparing to Non-Faces Dataset](#comparing-to-non-faces-dataset)
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
Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to extract important features from high-dimensional datasets. PCA works by identifying the principal components of the data, which are linear combinations of the original features that capture the most variation in the data.

#### Pseudo Code
- The pseudo code for the PCA:
```python
    # computing the mean
    means=np.mean(training_set,axis=0).reshape(1,10304)
    # centering the data
    centered_training_set=training_set-means
    # computing the covariance matrix
    covariance_matrix=np.cov(centered_training_set.T,bias=True)
    # computing the eigen vectors & eigen values
    eigenvalues,eigenvectors=np.linalg.eigh(covariance_matrix)
    
    # sorting eigen vectors according to their corresponding eigen values
    positions = eigenvalues.argsort()[::-1]
    
    sorted_eigenvectors = (eigenvectors[:,positions])
    
    total = sum(eigenvalues)
    
    # getting the required pcs to reach a certain alpha
    r = 0
    
    current_sum = 0

    while current_sum/total < alpha:
        current_sum += eigenvalues[r]
        r += 1
    # getting the new space that the data will be projected to it 
    new_space = eigenvectors[:, :r]   

    return new_space

```
### The first 2 Eigen-Faces

![image](https://user-images.githubusercontent.com/84376570/226112967-484b8bdd-6262-4c3d-b8c7-9457ce20ccd5.png)
![image](https://user-images.githubusercontent.com/84376570/226112979-b4425370-2c27-4b4e-8e2b-c4cd4d6c32b2.png)

##  Comparing different values of alpha to their corresponding accuracies

![image](https://user-images.githubusercontent.com/84376570/226113140-86c43a1e-192d-4224-b460-1992fd76757f.png)


## comparing different values of alpha to their corresponding number of principle components

![image](https://user-images.githubusercontent.com/84376570/226113163-8adc3be7-12bb-4005-8c62-703e595a8aef.png)

## comparing alpha to their corresponding accuracies after changing percentage of training split into 70% and test split into 30%
![image](https://user-images.githubusercontent.com/84376570/226113239-2459a38b-1924-4679-8bd6-1cb6800b8a7f.png)



## comparing alpha to their corresponding number of principle components after changing percentage of training split into 70% and test split into 30%

![image](https://user-images.githubusercontent.com/84376570/226113277-9f70238f-c133-48c5-9b5e-f7edeb73ba0e.png)

### Using K-NN Classifier after PCA
- KNN classifier is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output is determined by the majority of the classes of the k nearest neighbors.
- The following graph shows the accuracy of face recognition at different values of k (1-3-5-7)
![image](https://user-images.githubusercontent.com/84376570/226113360-f4b42806-89fd-4600-ba7c-e5e662754585.png)

### Comparison between different splitting ways
- this table shows difference in accuracies
- ![image](https://user-images.githubusercontent.com/84376570/226113668-e477a959-43d0-4fcb-8468-38b3c4d69b27.png)


- this table shows difference in number of principle components
![image](https://user-images.githubusercontent.com/84376570/226113679-1aa2a029-c65d-426f-bd6d-cd1949f05724.png)

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

#### Using K-NN Classifier after LDA
- KNN classifier is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output is determined by the majority of the classes of the k nearest neighbors.
- The following graph shows the accuracy of face recognition at different values of k (1-3-5-7)

![image](https://user-images.githubusercontent.com/41492875/226025723-23c4666d-f028-4d71-a146-f791358fd5e7.png)


## Comparing to Non-Faces Dataset
- We compared the results of the PCA and LDA algorithms to the results of the same algorithms on a non-faces dataset. The non-faces dataset is the []() dataset. The results are discussed inside the [notebook](faces_vs_nonfaces.ipynb) and in the [Results](#results) section.
## Results

## Contributers

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohammed Farid](https://github.com/MohamedFarid612)

- [Adham Mohammed](https://github.com/adhammohamed1)

