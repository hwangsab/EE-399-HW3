# EE-399: Introduction to Machine Learning
#### HW 3 Submission
#### Sabrina Hwang

## Abstract:
This code was created for EE 399 Introduction to Machine Learning, HW 3 submission by Sabrina Hwang. 
This code performs a variety of tasks related to analyzing and classifying the MNIST dataset of handwritten digits. The MNIST dataset is a classic dataset widely used for machine learning tasks and contains 70,000 grayscale images of size 28x28 pixels, with each image corresponding to a digit from 0 to 9.

Overall, this code provides a useful demonstration of how to perform an analysis of the MNIST dataset using SVD, as well as how to train and test classifiers to identify individual digits in the dataset. It also showcases several different classifiers and provides a comparison of their performance. This code could be used as a starting point for further analysis and classification tasks involving the MNIST dataset or other similar datasets.

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#code-description)
    * [Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization]()
    * [Part I Problem 2: Singular Value Spectrum and Rank Estimation]()
    * [Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis]()
    * [Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot]()
    * [Part II Problem (a): Linear classification of two digits using LDA]()
    * [Part II Problem (b): Linear classification of three digits using LDA]()
    * [Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers]()
    * [Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers]()
    * [Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers]()
    * [Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers]()
* [Computational Results](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#usage)
  * [Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization]()
  * [Part I Problem 2: Singular Value Spectrum and Rank Estimation]()
  * [Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis]()
  * [Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot]()
  * [Part II Problem (a): Linear classification of two digits using LDA]()
  * [Part II Problem (b): Linear classification of three digits using LDA]()
  * [Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers]()
  * [Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers]()
  * [Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers]()
  * [Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers]()
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
The first part of the code performs an analysis of the MNIST dataset using Singular Value Decomposition (SVD). The data is first scaled to the range [0, 1], and a random sample of 4000 images is taken. The images are then reshaped into column vectors and the SVD is performed on the centered data. The first 10 singular values are printed and the digit images corresponding to the first 10 columns of the centered data matrix are plotted. The singular value spectrum is also plotted to determine the number of modes necessary for good image reconstruction.

The code then moves onto building a classifier to identify individual digits in the training set. Two digits are chosen and a Linear Discriminant Analysis (LDA) classifier is trained to classify the digits. The data is split into training and testing sets, and several different classifiers are trained and tested, including Logistic Regression, Decision Trees, and Support Vector Machines (SVMs). The accuracy of each classifier is evaluated and compared to the accuracy of the LDA classifier.

## Theoretical Background:
The code is performing binary classification of the MNIST dataset using three different classifiers: Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Decision Tree. The objective of the code is to evaluate the performance of each classifier in separating pairs of digits from 0 to 9.

The MNIST dataset is a popular dataset in machine learning, consisting of 70,000 images of handwritten digits from 0 to 9, with 7,000 images for each digit. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. Each image is 28x28 pixels, and each pixel has a grayscale value between 0 and 255. The objective is to classify each image into one of the ten digit classes.

## Algorithm Implementation and Development:
This homework assignment works around the MNIST dataset loaded through the following lines of code:
```
mnist = fetch_openml('mnist_784', parser='auto')
X = mnist.data / 255.0  # Scale the data to [0, 1]
y = mnist.target
```

Completion of this project and subsequent development and implementation of the algorithm was 
accomplished through Python as our primary programming language. 

### Code Description
The code is written in Python and uses the following overarching libraries:  
* `numpy` for numerical computing  
* `matplotlib` for data visualization  
* `math` for mathematical functions  
* `random` for random generation
* `scipy` for regression

The code also uses the following libraries for Part I:
* from `sklearn.datasets` import `fetch_openml`
* from `sklearn.decomposition` import `PCA`
* import `matplotlib.pyplot` as `plt`
* from `mpl_toolkits.mplot3d` import `Axes3D`
* from `scipy.io` import `loadmat`
* from `scipy.sparse.linalg` import `eigs`
* from `numpy` import `linalg`
  
And the following additional libraries for Part II:
* from `sklearn.discriminant_analysis` import `LinearDiscriminantAnalysis`
* from `sklearn.model_selection` import `train_test_split`
* from `sklearn.linear_model` import `LogisticRegression`
* from `sklearn.metrics` import `accuracy_score`
* from `sklearn.tree` import `DecisionTreeClassifier`
* from `sklearn.svm` import `SVC`

#### Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization
The first problem involves performing an SVD (Singular Value Decomposition) analysis of the digit images. The images are first reshaped into column vectors. A random sample of 4000 images is extracted for this problem. The SVD of the centered data is then computed using the numpy `linalg.svd()` function, and the first 10 singular values and their corresponding digit images are printed.

```
images = X[:, :100]
C = np.matmul(images.T, images)
```
#### Part I Problem 2: Singular Value Spectrum and Rank Estimation
The second problem involves finding the number of modes (rank r of the digit space) necessary for good image reconstruction by analyzing the singular value spectrum. The SVD is performed on the full dataset, and the index of the first singular value that explains at least 90% of the total variance is found. The proportion of total variance explained by each singular value is then computed and plotted to show the singular value spectrum.

```
most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])
```
#### Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis
The third problem asks for the interpretation of the U, Σ, and V matrices in SVD. There is no explicit code to answer this problem.

```
images = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
image_list = X[:, np.subtract(images, 1)]

C = np.ndarray((10, 10))
C = np.matmul(image_list.T, image_list)
```
#### Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot
The fourth problem involves projecting the images onto three selected V-modes (columns) colored by their digit label on a 3D plot. For this problem, the MNIST data is again loaded, and PCA (Principal Component Analysis) is performed on the data using the `PCA` function from `sklearn.decomposition`. The second, third, and fifth principal components are selected, and the 3D scatter plot is created using the `mpl_toolkits.mplot3d` module.

```
Y = np.dot(X, X.T)
eigenvalues, eigenvectors = np.linalg.eigh(Y)

W = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, W]

v_1 = eigenvectors[:, 0]
```
#### Part II Problem (a): Linear classification of two digits using LDA
This problem involves the selection of two digits (3 and 8) from the dataset and tries to build a linear classifier to classify them. It first selects only the data samples for these two digits and applies LDA to reduce the dimensionality of the data. It then trains a logistic regression classifier on the transformed data and evaluates its accuracy on a test set.

```
U, S, V = np.linalg.svd(X, full_matrices = False)
first_six = V[:6, :]
```
#### Part II Problem (b): Linear classification of three digits using LDA
Problem (b) is an extension of problem (a), only now the code selects three digits (3, 7, and 8) from the dataset and tries to build a linear classifier to classify them. It follows the same process as in section (a) to prepare the data and train a classifier.

```
u_1 = U[:, 0]
norm_of_difference = np.linalg.norm(np.abs(v_1) - np.abs(u_1))
```
#### Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers
This problem compares all pairs of digits in the dataset to determine which pair is most difficult to separate. It calculates the accuracy of the LDA classifier on the test set for each pair of digits and stores the results in a dictionary.

#### Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers
This problem asks to determine the two digits that are the easiest to separate, the code compares all pairs of digits and stores their accuracy using LDA for dimensionality reduction and Logistic Regression for classification. The digit pair with the highest accuracy is considered the easiest to separate.

#### Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers
Similarly to problem (d) and problem (e), this problem asks for computation of the two digits easiest to separate, as well as the two digits most difficult to separate, using SVM and decision tree classifiers. 

#### Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers
This problem asks to compare the performance between LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate. There is also no explicit code to answer this problem.

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW3.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Part I Problem 1: SVD Analysis of Digit Images with Random Sampling and Visualization
The resultant dot product (correlation) matrix between the first 100 images are plotted as followed:  
![download](https://user-images.githubusercontent.com/125385468/232674818-baf7ce66-d67c-465b-96e6-94afe61e22dc.png)

#### Part I Problem 2: Singular Value Spectrum and Rank Estimation
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part I Problem 3: Interpretation of U, Σ, and V Matrices in SVD Analysis
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part I Problem 4: Visualization of selected V-modes of PCA with 3D scatter plot
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (a): Linear classification of two digits using LDA
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (b): Linear classification of three digits using LDA
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (c): Identifying the most difficult digit pairs to separate using LDA classifiers
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (d): Identifying the most easy digit pairs to separate using LDA classifiers
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (e): Identifying most easy and difficult digit pairs using SVM and decision tree classifiers
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

#### Part II Problem (f): Comparing the performance between LDA, SVM, and decision tree classifiers
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
stuff
```

In addition, plotted SVD nodes were as followed:  
![download](https://user-images.githubusercontent.com/125385468/232675682-a428b9fe-b24c-483a-9631-1cf849293455.png)

## Summary and Conclusions:
stuff here

Overall, this assignment provided a comprehensive analysis of the MNIST dataset using various classification techniques, and evaluated their performance on different pairs of digits. It demonstrated the importance of feature selection and the power of different classifiers for identifying patterns in data.
