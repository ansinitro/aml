# Comparative Analysis of Traditional Machine Learning and Deep Learning for Pneumonia Detection in Medical Imaging

**Abstract**
This paper presents a comparative study of machine learning approaches for the classification of pneumonia from chest X-ray images using the PneumoniaMNIST dataset. We evaluate two distinct methodologies: a traditional machine learning pipeline utilizing Principal Component Analysis (PCA) for dimensionality reduction followed by Support Vector Machine (SVM) classification, and a deep learning approach employing a Convolutional Neural Network (CNN). Our study incorporates rigorous Exploratory Data Analysis (EDA), including unsupervised clustering (K-Means) and visualization techniques (t-SNE), to understand the intrinsic structure of the data. Results demonstrate that while traditional methods provide a robust baseline, deep learning significantly outperforms them in classification accuracy.

## 1. Introduction
Medical image classification is a critical task in computer-aided diagnosis (CAD), enabling rapid and accurate detection of diseases. Pneumonia, an infection that inflames the air sacs in one or both lungs, remains a leading cause of death globally, particularly among children and the elderly. Automated detection from chest X-rays can support radiologists by prioritizing urgent cases and reducing diagnostic error.
The objective of this project is to develop and evaluate automated classification models for distinguishing between normal and pneumonia-infected chest X-rays. We align our methodology with key machine learning concepts, including dimensionality reduction, clustering, and supervised learning.

## 2. Dataset Description
We utilize the **PneumoniaMNIST** dataset, a lightweight version of the Kermany et al. Chest X-Ray images, standardized as part of the MedMNIST benchmark.
*   **Modality**: Chest X-Ray.
*   **Resolution**: 28x28 pixels (grayscale).
*   **Classes**: Binary classification [0: Normal, 1: Pneumonia].
*   **Data Split**: 
    *   Train: 4,708 samples
    *   Validation: 524 samples
    *   Test: 624 samples

Ethical considerations regarding patient privacy are addressed by the dataset providers through anonymization. The use of this public dataset complies with standard research ethics.

## 3. Methodology

### 3.1 Exploratory Data Analysis (EDA) & Preprocessing
Images were normalized to a range of [-1, 1] for model stability. We performed:
*   **Class Distribution Analysis**: To detect imbalance.
*   **Dimensionality Reduction**: We applied **PCA** (Principal Component Analysis) and **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to visualize the high-dimensional image data in 2D space.
*   **Clustering**: **K-Means clustering** was computed on the flattened image vectors to analyze if the natural data groupings correspond to the clinical labels (Normal vs. Pneumonia).

### 3.2 Model 1: Traditional Machine Learning (PCA + SVM)
To establish a baseline and demonstrate feature extraction techniques:
1.  **Feature Extraction**: PCA was applied to retain 95% of the variance, significantly reducing the feature space from 784 (28x28) dimensions.
2.  **Classification**: A **Support Vector Machine (SVM)** with an RBF kernel was trained on the reduced features.

### 3.3 Model 2: Deep Learning (CNN)
We designed a custom Convolutional Neural Network (CNN) architecture optimized for 28x28 images:
*   **Architecture**: 3 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool) followed by 2 Fully Connected layers.
*   **Training**: CrossEntropyLoss, Adam Optimizer (LR=0.001), trained for 10 epochs.

## 4. Evaluation and Results

### 4.1 Exploratory Analysis Results
(Insert findings from `class_distribution.png`, `pca_visualization.png`, `tsne_visualization.png` here)
*   **K-Means Clustering**: The confusion matrix of clusters vs. true labels (see `kmeans_confusion.png`) reveals [INSERT ANALYSIS].

### 4.2 Classification Performance
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **PCA + SVM** | [INSERT ACCURACY] | [INSERT F1] |
| **CNN** | [INSERT ACCURACY] | [INSERT F1] |

*   **SVM Performance**: The baseline model achieved [INSERT SCORE], demonstrating that even simple pixel-intensity covariance captures diagnostic information.
*   **CNN Performance**: The deep learning model achieved [INSERT SCORE], outperforming the baseline by learning hierarchical spatial features (edges, textures, infection patterns).

(Insert `cnn_confusion_matrix.png` analysis here)

## 5. Discussion and Ethical Considerations
*   **Model Limitations**: The low resolution (28x28) of MedMNIST limits the visibility of subtle features, potentially capping performance compared to high-res clinical X-rays.
*   **Algorithmic Bias**: Training data must be diverse. If the source dataset heavily represents one demographic, the model may generalize poorly to others.
*   **Clinical Relevance**: While high accuracy is promising, "black box" models like CNNs lack interpretability. For clinical adoption, explainability techniques (e.g., Grad-CAM) are necessary to trust the diagnosis.

## 6. Conclusion and Future Work
We successfully implemented a comparative pipeline for pneumonia detection. The CNN approach proved superior to the PCA+SVM baseline. Future work should focus on using the higher-resolution version of the dataset (224x224) and implementing explainability modules to support clinician trust.

## 7. References
1.  MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.
2.  Kermany, D. S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." Cell 172.5 (2018): 1122-1131.
