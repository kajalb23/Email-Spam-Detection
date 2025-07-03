# Email Spam Classification Using Logistic Regression


This project aims to develop a binary classification model that predicts whether an email is spam or not based on the frequency of commonly occurring words. A logistic regression model is used due to its interpretability and effectiveness for linearly separable classification problems. The project follows a systematic machine learning pipeline including data preprocessing, model training, evaluation, and performance reporting.


## Dataset Overview :
Source: Provided CSV file containing pre-processed word frequency data from emails.

Total Samples: 5,172 emails


## Features:

Columns 1 to 3000: Word count features (frequency of top 3000 words per email)

Column 0: Email ID (anonymized)

Column 3001: Target label (1 = Spam, 0 = Not Spam)

The dataset represents each email as a feature vector of word frequencies, allowing for compact storage and efficient modeling without textual parsing or NLP pipelines.


## Objective :
To build a machine learning model that can accurately classify emails as spam or not spam using logistic regression, and evaluate its performance using appropriate classification metrics.


## Workflow :
Data Loading and Exploration

Loaded the dataset using pandas and performed basic checks on shape, types, and missing values.


## 1. Data Preprocessing

Extracted feature matrix X and target vector Y.

Applied feature scaling using StandardScaler to normalize data for logistic regression.


## 2. Model Training

Split the dataset into training and testing sets using an 80:20 ratio.

Trained a logistic regression model using scikit-learn with increased iterations to ensure convergence.

## 3. Model Evaluation

Generated predictions on the test set.

Evaluated performance using confusion matrix, accuracy, precision, recall, and F1-score.

Interpreted model results and summarized them in tabular form.

### Model Performance
Based on evaluation on the test set:

| **Metric**   | **Value** |
|--------------|-----------|
| Accuracy     | 96.52%    |
| Precision    | 90.37%    |
| Recall       | 98.31%    |
| F1 Score     | 93.90%    |


The model demonstrated strong overall performance with high recall, meaning it effectively detected spam emails.

Precision was slightly lower, indicating a small number of legitimate emails were misclassified as spam.

The confusion matrix confirms minimal false negatives and manageable false positives.

## 4. Confusion Matrix:

| **Actual \\ Predicted** | **Not Spam (0)** | **Spam (1)** |
|-------------------------|------------------|--------------|
| **Not Spam (0)**        | 708              | 31           |
| **Spam (1)**            | 5                | 291          |


## 5. Technologies Used :
1. Python 3.9+
2. Pandas – data manipulation
3. NumPy – numerical computing
4. Scikit-learn – machine learning model and metrics
5. Matplotlib, Seaborn – visualization
6. Google Colab / Jupyter Notebook – development environment

