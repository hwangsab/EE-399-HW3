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
    * [Problem (a): Computing Correlation Matrix using Dot Product]()
    * [Problem (b): Identifying Highly Correlated and Uncorrelated Images]()
    * [Problem (c): Computing Correlation Matrix for Subset of Images]()
    * [Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$]()
    * [Problem (e): Finding the First Six Principal Component Directions using SVD]()
    * [Problem (f): Comparing First Eigenvector and First SVD Mode]()
    * [Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them]()
* [Computational Results](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399-HW3/blob/main/README.md#usage)
  * [Problem (a): Computing Correlation Matrix using Dot Product]()
  * [Problem (b): Identifying Highly Correlated and Uncorrelated Images]()
  * [Problem (c): Computing Correlation Matrix for Subset of Images]()
  * [Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$]()
  * [Problem (e): Finding the First Six Principal Component Directions using SVD]()
  * [Problem (f): Comparing First Eigenvector and First SVD Mode]()
  * [Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them]()
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

#### Part I Problem 1: 
The first problem involves performing an SVD (Singular Value Decomposition) analysis of the digit images. The images are first reshaped into column vectors. A random sample of 4000 images is extracted for this problem. The SVD of the centered data is then computed using the numpy `linalg.svd()` function, and the first 10 singular values and their corresponding digit images are printed.

```
images = X[:, :100]
C = np.matmul(images.T, images)
```
#### Part I Problem 2: 
The second problem involves finding the number of modes (rank r of the digit space) necessary for good image reconstruction by analyzing the singular value spectrum. The SVD is performed on the full dataset, and the index of the first singular value that explains at least 90% of the total variance is found. The proportion of total variance explained by each singular value is then computed and plotted to show the singular value spectrum.

```
most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])
```
#### Part I Problem 3: 
The third problem asks for the interpretation of the U, Σ, and V matrices in SVD. There is no explicit code to answer this problem.

```
images = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]
image_list = X[:, np.subtract(images, 1)]

C = np.ndarray((10, 10))
C = np.matmul(image_list.T, image_list)
```
#### Part I Problem 4: 
The fourth problem involves projecting the images onto three selected V-modes (columns) colored by their digit label on a 3D plot. For this problem, the MNIST data is again loaded, and PCA (Principal Component Analysis) is performed on the data using the `PCA` function from `sklearn.decomposition`. The second, third, and fifth principal components are selected, and the 3D scatter plot is created using the `mpl_toolkits.mplot3d` module.

more stuff here

```
Y = np.dot(X, X.T)
eigenvalues, eigenvectors = np.linalg.eigh(Y)

W = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, W]

v_1 = eigenvectors[:, 0]
```
#### Part II Problem (a): Finding the First Six Principal Component Directions using SVD
This problem involves the selection of two digits (3 and 8) from the dataset and tries to build a linear classifier to classify them. It first selects only the data samples for these two digits and applies LDA to reduce the dimensionality of the data. It then trains a logistic regression classifier on the transformed data and evaluates its accuracy on a test set.

```
U, S, V = np.linalg.svd(X, full_matrices = False)
first_six = V[:6, :]
```
#### Part II Problem (b): Comparing First Eigenvector and First SVD Mode
Problem (b) is an extension of problem (a), only now the code selects three digits (3, 7, and 8) from the dataset and tries to build a linear classifier to classify them. It follows the same process as in section (a) to prepare the data and train a classifier.

```
u_1 = U[:, 0]
norm_of_difference = np.linalg.norm(np.abs(v_1) - np.abs(u_1))
```
#### Part II Problem (c): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
This problem compares all pairs of digits in the dataset to determine which pair is most difficult to separate. It calculates the accuracy of the LDA classifier on the test set for each pair of digits and stores the results in a dictionary.

#### Part II Problem (d): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
This problem asks to determine the two digits that are the easiest to separate, the code compares all pairs of digits and stores their accuracy using LDA for dimensionality reduction and Logistic Regression for classification. The digit pair with the highest accuracy is considered the easiest to separate.

#### Part II Problem (e): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
Similarly to problem (d) and problem (e), this problem asks for computation of the two digits easiest to separate, as well as the two digits most difficult to separate, using SVM and decision tree classifiers. 

## Computational Results:

### Usage
To run the code, simply run the Python file `EE399 HW3.py` in any Python environment. The output will be 
printed to the console and displayed in a pop-up window.

#### Problem (a): Computing Correlation Matrix using Dot Product
The resultant dot product (correlation) matrix between the first 100 images are plotted as followed:  
![download](https://user-images.githubusercontent.com/125385468/232674818-baf7ce66-d67c-465b-96e6-94afe61e22dc.png)

#### Problem (b): Identifying Highly Correlated and Uncorrelated Images
Using the first part of the program to determine the pairs of highly correlated and uncorrelated images: 
```
most_correlated = np.argwhere(C == np.sort(C.flatten())[-3])
least_correlated = np.argwhere(C == np.sort(C.flatten())[1])

print (most_correlated[0], least_correlated[0])
```
We could determine that the pairs `[86 88]` and `[54 64]` represent the indices of the images we are looking for. 

Plotting the following images yield:  
![download](https://user-images.githubusercontent.com/125385468/232675169-a2d1cddd-13b8-47db-b0f5-4176f676c089.png)

and  
![download](https://user-images.githubusercontent.com/125385468/232675206-9cf1c489-db47-4a41-a928-71bfea5d6583.png)

#### Problem (c): Computing Correlation Matrix for Subset of Images
The resultant dot product (correlation) matrix between the first 10 images are plotted as followed:  
![download](https://user-images.githubusercontent.com/125385468/232675344-552f1047-e830-4f7a-b9a7-c3a4ddbce12c.png)

#### Problem (d): Finding the First Six Eigenvectors of $Y=XX^T$
The resultant 6 eigenvectors with the largest magnitude eigenvalue are determined to be:
```
[[-0.02384327  0.04535378 -0.05653196 ... -0.00238077  0.0015886
  -0.00041024]
 [-0.02576146  0.04567536 -0.04709124 ...  0.00265168 -0.00886967
   0.0047811 ]
 [-0.02728448  0.04474528 -0.0362807  ... -0.00073077  0.00706009
  -0.00678472]
 ...
 [-0.02082937 -0.03737158 -0.06455006 ... -0.0047683  -0.00596037
  -0.0032901 ]
 [-0.0193902  -0.03557383 -0.06196898 ... -0.00173228 -0.00175508
  -0.00131795]
 [-0.0166019  -0.02965746 -0.05241684 ...  0.00458062  0.00266653
   0.00168849]]
```

#### Problem (e): Finding the First Six Principal Component Directions using SVD
The first six principal component directions are determined to be:
```
[[-0.01219331 -0.00215188 -0.01056679 ... -0.02177117 -0.03015309
  -0.0257889 ]
 [-0.01938848 -0.00195186  0.02471869 ...  0.04027773  0.00219562
   0.01553129]
 [ 0.01691206  0.00143586  0.0384465  ...  0.01340245 -0.01883373
   0.00643709]
 [ 0.0204079  -0.01201431  0.00397553 ... -0.01641295 -0.04011563
   0.02679029]
 [-0.01902342  0.00418948  0.0384026  ... -0.01092512  0.00087341
   0.01260435]
 [-0.0090084  -0.00624237  0.01580824 ... -0.00977639  0.00090316
   0.00304479]]
```

#### Problem (f): Comparing First Eigenvector and First SVD Mode
The norm of the difference of absolute values between `v_1` and `u_1` was determined to be:
```
7.394705201660225e-16
```

#### Problem (g): Computing Variance Captured by Each of the First 6 SVD Modes and Plotting Them
Computing the percentage of variance captured by each of the first 6 SVD modes yielded:
```
Percentage of variance captured by each SVD mode 1: 72.93%
Percentage of variance captured by each SVD mode 2: 15.28%
Percentage of variance captured by each SVD mode 3: 2.57%
Percentage of variance captured by each SVD mode 4: 1.88%
Percentage of variance captured by each SVD mode 5: 0.64%
Percentage of variance captured by each SVD mode 6: 0.59%
```

In addition, plotted SVD nodes were as followed:  
![download](https://user-images.githubusercontent.com/125385468/232675682-a428b9fe-b24c-483a-9631-1cf849293455.png)

## Summary and Conclusions:
stuff here

Overall, this assignment provided a comprehensive analysis of the MNIST dataset using various classification techniques, and evaluated their performance on different pairs of digits. It demonstrated the importance of feature selection and the power of different classifiers for identifying patterns in data.
