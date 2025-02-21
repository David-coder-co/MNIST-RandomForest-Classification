# MNIST-RandomForest-Classification

This project utilizes the MNIST dataset to build and evaluate an image classification model using RandomForestClassifier. The goal is to preprocess the image data, train the model, tune hyperparameters, and assess performance based on multiple metrics.

### Project Overview

The project focuses on the MNIST dataset, which contains images of handwritten digits. The goal is to develop a classification model that can accurately identify these digits using the Random Forest algorithm. The project involves preprocessing the data, training the model, and tuning key hyperparameters to optimize performance. Additionally, the model's effectiveness is evaluated through various metrics such as accuracy, precision, recall, and F1-score, with a particular emphasis on understanding the model's performance across different classes of digits.

### Methodology
- Data Loading: The MNIST dataset is loaded using the sklearn.datasets library.
- Data Split: The dataset is divided into training and test sets. The purpose is to train the model on one set (training) and evaluate its performance on another (test).
- Model Training: The RandomForestClassifier is used for training the model.
- Hyperparameter Tuning: One parameter selected for tuning is n_estimators, which controls the number of trees in the Random Forest. By adjusting this parameter, the effect on model performance is evaluated. Increasing the number of trees typically improves the model's accuracy but also increases computational complexity.

### Performance Evaluation:
- Confusion Matrix
- Accuracy, Precision, Recall, F1-Score (macro average)
Insight: Identify which classes the model struggles with the most.

### Installation
- Clone the repository to your local environment.
- Install the required dependencies: refer to requirements.txt

### Usage
Run the analysis script to load the dataset, train the model, and print evaluation metrics: python mnist_task.py

### Results
- Confusion matrix and evaluation metrics (accuracy, precision, recall, F1-score) will be printed for model performance.
- Key insights on model behavior, including struggles with certain digit classes, will be displayed.
