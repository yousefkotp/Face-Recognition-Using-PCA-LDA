# Face Recognition
A face recognition project using PCA and LDA algorithms.

> This readme file is a summary of the project. For more details, please refer to the [notebook](faces_vs_nonfaces.ipynb).
## Table of Contents
- [Face Recognition](#face-recognition)
  * [Table of Contents](#table-of-contents)
  * [Dataset](#dataset)
    + [Data Splitting](#data-splitting)
  * [Algorithms](#algorithms)
    + [PCA](#pca)
      - [Pseudo Code](#pseudo-code)
      - [The first 2 Eigen-Faces](#the-first-2-eigen-faces)
      - [Comparing different values of alpha to their corresponding accuracies](#comparing-different-values-of-alpha-to-their-corresponding-accuracies)
      - [Comparing different values of alpha to their corresponding number of principle components](#comparing-different-values-of-alpha-to-their-corresponding-number-of-principle-components)
      - [Comparing alpha to their corresponding accuracies after changing percentage of training split into 70% and test split into 30%](#comparing-alpha-to-their-corresponding-accuracies-after-changing-percentage-of-training-split-into-70--and-test-split-into-30-)
      - [Comparing alpha to their corresponding number of principle components after changing percentage of training split into 70% and test split into 30%](#comparing-alpha-to-their-corresponding-number-of-principle-components-after-changing-percentage-of-training-split-into-70--and-test-split-into-30-)
      - [Using K-NN Classifier after PCA](#using-k-nn-classifier-after-pca)
      - [Comparison between different splitting ways](#comparison-between-different-splitting-ways)
      - [Using PCA Variations](#using-pca-variations)
        * [Randomized PCA](#randomized-pca)
        * [Kernel PCA](#kernel-pca)
      - [Figure showing accuracies for PCA variations](#figure-showing-accuracies-for-pca-variations)
    + [LDA](#lda)
      - [Pseudo Code](#pseudo-code-1)
      - [Using K-NN Classifier after LDA](#using-k-nn-classifier-after-lda)
      - [Comparison between different splitting ways](#comparison-between-different-splitting-ways-1)
      - [Using LDA Variations](#using-lda-variations)
        * [Kernel LDA](#kernel-lda)
      - [Figure showing accuracies for LDA variations](#figure-showing-accuracies-for-lda-variations)
  * [Comparing to Non-Faces Dataset](#comparing-to-non-faces-dataset)
  * [Contributers](#contributers)


## Dataset
- Our dataset for this project is the [AT&T Face Database](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces). The dataset is open-source and can be downloaded from Kaggle.
- The dataset contains 400 images of 40 people. Each person has 10 images. The images are of size 92x112 pixels. The images are in grayscale. The images are in the form of a numpy array. The images are stored in the `archive` folder.

### Data Splitting
- We tried splitting the dataset into training and testing sets. one split was 50-50 and the other one is 70-30. The results are discussed inside the [notebook](faces_vs_nonfaces.ipynb).

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
#### The first 2 Eigen-Faces

![image](https://user-images.githubusercontent.com/84376570/226113767-05d70c78-6322-46bd-9952-c6d1c7a2bf2c.png)
![image](https://user-images.githubusercontent.com/84376570/226113790-6725cba6-696b-4edc-9b90-b6f745411572.png)
####  Comparing different values of alpha to their corresponding accuracies
- this figure shows that accuracy increases as alpha increases until a certain limit.
![image](https://user-images.githubusercontent.com/84376570/226113140-86c43a1e-192d-4224-b460-1992fd76757f.png)


#### Comparing different values of alpha to their corresponding number of principle components
- this figure shows that number of principle components increase as alpha increases.
![image](https://user-images.githubusercontent.com/84376570/226113163-8adc3be7-12bb-4005-8c62-703e595a8aef.png)

#### Comparing alpha to their corresponding accuracies after changing percentage of training split into 70% and test split into 30%
- this figure shows that accuracy increases as alpha increases.
- it is obvious that the curve has the same shape as the last splits but with higher values as the training data has increased.
![image](https://user-images.githubusercontent.com/84376570/226113239-2459a38b-1924-4679-8bd6-1cb6800b8a7f.png)



#### Comparing alpha to their corresponding number of principle components after changing percentage of training split into 70% and test split into 30%
- this figure shows that number of principle components increase as alpha increases until a certain limit.
- it is obvious that the curve has the same shape as the last splits but with higher values as the training data has increased
so the number of principle components needed to handle the same total percentage of the variance increased.
![image](https://user-images.githubusercontent.com/84376570/226113277-9f70238f-c133-48c5-9b5e-f7edeb73ba0e.png)

#### Using K-NN Classifier after PCA
- KNN classifier is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output is determined by the majority of the classes of the k nearest neighbors.
- The following graph shows the accuracy of face recognition at different values of k (1-3-5-7)
![image](https://user-images.githubusercontent.com/84376570/226113360-f4b42806-89fd-4600-ba7c-e5e662754585.png)

#### Comparison between different splitting ways
- this table shows difference in accuracies

![image](https://user-images.githubusercontent.com/84376570/226113668-e477a959-43d0-4fcb-8468-38b3c4d69b27.png)


- this table shows difference in number of principle components
![image](https://user-images.githubusercontent.com/84376570/226113679-1aa2a029-c65d-426f-bd6d-cd1949f05724.png)

#### Using PCA Variations

##### Randomized PCA
  - Randomized PCA is a faster and more memory-efficient version of PCA that uses randomized matrix approximations to estimate the principal components of the data. This approach involves sampling subsets of the data and computing the eigenvectors of the resulting covariance matrix, which can be done more efficiently than computing the eigenvectors of the full covariance matrix.
  - the randomised version of PCA operates in O(nd^2) + O(d^3) where d is the number of principle components, conventional PCA operates in O(np^2) + O(p^3) where n is the number of data points and p is the number of features. Therefore, it moves extremely quickly when d is significantly smaller than n.

##### Kernel PCA
  - Kernel PCA is a non-linear dimensionality reduction technique that uses a kernel function to map high-dimensional data into a lower-dimensional space. This allows it to capture non-linear relationships between variables that are not possible with linear PCA.
  - The time complexity of normal PCA is O(d^3), where d is the number of dimensions, while the time complexity of kernel PCA is O(n^3), where n is the number of data points. The computation of the kernel matrix is the most computationally expensive step in kernel PCA.
  - Kernel PCA may be more accurate than normal PCA for datasets with non-linear relationships between variables, as it can capture these relationships. However, kernel PCA is more prone to overfitting than normal PCA, and the choice of kernel function can greatly affect the performance of kernel PCA.

#### Figure showing accuracies for PCA variations
- Kernel PCA, specifically using the radial basis function (RBF) kernel, may fail when the dataset has a large number of dimensions or when the number of data points is much larger than the number of dimensions. This is because the kernel matrix can become very large and computationally expensive to compute and manipulate. Additionally, the choice of kernel function and its parameters can greatly affect the performance of kernel PCA. In contrast, normal PCA may perform better in high-dimensional datasets or when the relationships between variables are linear, as it is designed to capture linear relationships between variables.
- RBF kernel PCA uses the radial basis function kernel, which is a Gaussian function that measures the distance between data points in the original space. This kernel is useful for capturing non-linear relationships between variables that cannot be captured by linear PCA.While Polynomial kernel PCA uses a polynomial kernel, which is a power function that measures the dot product between data points in the original space raised to a certain power. This kernel is also useful for capturing non-linear relationships between variables, but is more sensitive to outliers and noise than the RBF kernel.

- ![image](https://user-images.githubusercontent.com/84376570/226116045-eb0d999c-74f4-413a-8fb0-497f218c7dbc.png)

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

#### Comparison between different splitting ways
- We Tried splitting the data into training and test in 2 different ways:
1. 50-50: This splitting resulted in a good accuracy reaching 95.5%
2. 70-30: This splitting resulted in a slightly better accuracy reaching 95.83%

#### Using LDA Variations
- We tried using different variations of LDA to see if we can get better results. The variations we tried are:
##### Shrinkage LDA
- Shrinkage LDA (Linear Discriminant Analysis) is a variant of the standard LDA method that is used for classification and dimensionality reduction. The key difference between shrinkage LDA and normal LDA is that the former incorporates a regularization term that shrinks the sample covariance matrix towards a diagonal matrix.

- This regularization is particularly useful when dealing with high-dimensional data, as it helps to overcome the small sample size problem by stabilizing the covariance estimates. Shrinkage LDA has been shown to outperform traditional LDA in terms of classification accuracy, especially when the number of features is much larger than the number of observations.

- Another advantage of shrinkage LDA is that it can handle multicollinearity between the predictor variables, which can be a problem in standard LDA when the predictors are highly correlated. In summary, shrinkage LDA is a powerful tool for classification and dimensionality reduction that can improve the accuracy of LDA in high-dimensional and small sample size settings.

#### Figure showing accuracies for LDA variations

![image](https://user-images.githubusercontent.com/41492875/226120140-bc599977-d70a-4b75-8b53-cdb52f042014.png)

## Comparing to Non-Faces Dataset
- We compared the results of the PCA and LDA algorithms to the results of the same algorithms on a non-faces dataset. The non-faces dataset is the []() dataset. The results are discussed inside the [notebook](faces_vs_nonfaces.ipynb).

## Faces VS Non-Faces Solutions__

1. Success & failure cases figure
   - PCA
   - ![image](https://user-images.githubusercontent.com/84376570/226126375-a50263e2-d5c9-40b4-9497-88e8a0387e91.png)
   - LDA
   - ![image](https://user-images.githubusercontent.com/84376570/226126647-0dfb68e5-6ff9-4e15-9b28-758d04d1e4cf.png)


2. We will use 1 dominant eigenvector for LDA as we have 2 classes
3. Accuracy vs number of non-face images figure
    - PCA
    - ![image](https://user-images.githubusercontent.com/84376570/226126339-7bb42742-0965-4453-8074-2fa0bb3de6e1.png)
    - LDA
    - ![image](https://user-images.githubusercontent.com/84376570/226126627-0497f70d-5e22-4934-9fd5-71f7a9039f58.png)

4. As the number of non-face images increases, the accuracy of the classifier decreases, in contrast to what might be expected. This is because the number of points in the space increases, and the K-NN classifier is more likely to be confused by the noise in the data. This is a limitation of the classifier, and is not a problem with the data. The noise causes the space to be more complex and the gaps between the classes to be smaller and smaller which is more likely to cause confusion for the K-NN classifier.
  


## Contributers

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohamed Farid](https://github.com/MohamedFarid612)

- [Adham Mohammed](https://github.com/adhammohamed1)

