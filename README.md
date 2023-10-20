# HASYv2_classifier
The project is a reupload of a project done in between March 2023 - June 2023.

This paper details our approach to classifying handwritten mathematical symbols using a range of machine learning techniques. We implemented and assessed various machine learning methods using a carefully selected subset of the HASYv2 dataset, which comprises 32x32 images of hand-drawn mathematical symbols.

To achieve symbol classification, we employed a variety of techniques:
#### K-Means Clustering 

Objective: Initially, K-Means serves as a data preprocessing tool.

Explanation: Handwritten symbols with similar characteristics are grouped into clusters based on their feature vectors. This aids in reducing the dataset's dimensionality and can support tasks like data labeling or outlier detection.

#### Logistic Regression

Objective: Logistic regression is utilized for binary classification tasks.

Explanation: It can classify symbols into one of two categories, such as distinguishing between "0" and "1" in handwritten digits. Logistic regression is computationally efficient and provides probability outputs, aiding in decision-making.

#### Support Vector Machines (SVM)

Objective: SVM is suitable for both binary and multi-class classification.

Explanation: SVM effectively separates distinct handwritten symbols in a high-dimensional space by identifying a hyperplane that maximizes the margin between classes. It proves valuable when dealing with complex symbol recognition tasks.

#### Multi-Layer Perception (MLP)

Objective: MLP is a versatile choice for deep learning-based symbol classification.
Explanation: MLPs are artificial neural networks with multiple hidden layers, capable of capturing intricate patterns and relationships within handwritten symbols. They are well-suited for tasks where features are hierarchical or nonlinear.

#### Convolutional Neural Network (CNN)  

Objective: CNNs excel in image-based classification tasks.
Explanation: CNNs are specifically designed for processing images and automatically learn relevant features from handwritten symbol images. They are highly effective at recognizing patterns and shapes in symbols due to their convolutional and pooling layers.

#### Principal Component Analysis (PCA) 

Objective: PCA is employed for dimensionality reduction.
Explanation: PCA helps reduce the computational load and enhance the efficiency of other algorithms by decreasing the number of features while preserving as much variance as possible. This is particularly useful when handling a large number of features associated with handwritten symbols.

**Please see reports of the project**

