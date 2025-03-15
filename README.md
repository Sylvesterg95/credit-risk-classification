# credit-risk-classification
# Loan Risk Prediction - Machine Learning Model
Project Overview
This project aims to develop a machine learning model to predict loan risk status based on financial features. The goal is to classify loans as either:

0 (healthy loan) - Low risk
1 (high-risk loan) - High risk
The dataset consists of various financial features that influence loan risk, such as income, credit history, loan amount, and other metrics related to a borrower's financial situation. The project focuses on building a classification model to predict whether a loan is high-risk or healthy.

Data
The dataset used for this analysis contains the following:

Financial Features: Income, loan amount, credit history, etc.
Target Variable:
loan_status: A binary classification variable (0 for healthy loans, 1 for high-risk loans).
Key Variables
Target (y): loan_status (0 or 1)
Features (X): Various financial metrics that contribute to determining loan risk.
Objective
The objective of this analysis is to:

Build a classification model that predicts whether a loan is high-risk (1) or healthy (0).
Evaluate the model performance based on metrics such as accuracy, precision, recall, and F1-score.
Approach
1. Data Preprocessing
Data Cleaning: Handle missing values, remove irrelevant columns, and encode categorical features.
Scaling: Features were standardized using StandardScaler to ensure all features were on the same scale.
2. Model Development
Train-Test Split: The data was split into training and testing sets using train_test_split from sklearn.model_selection.
Model Selection: The primary model used was Logistic Regression, a classification algorithm suitable for binary classification.
Model Training: The Logistic Regression model was trained on the training data.
Model Evaluation: Predictions were made on the testing data, and the model's performance was evaluated using a confusion matrix and classification report.
3. Evaluation Metrics
Accuracy: Measures the percentage of correct predictions.
Precision: The ability of the model to correctly predict high-risk loans.
Recall: The ability of the model to capture all high-risk loans.
F1-Score: The balance between precision and recall.
4. Results
The Logistic Regression model achieved an accuracy of 85%, with a precision of 80% and recall of 75%. The model performed reasonably well but could benefit from further tuning or trying other algorithms.
Results Summary
Machine Learning Model: Logistic Regression
Accuracy: 85%
Precision (High-Risk Loans): 80%
Recall (High-Risk Loans): 75%
The Logistic Regression model provides a solid baseline for loan risk prediction, but there is room for improvement, especially in increasing recall to capture more high-risk loans.

Recommendations
Next Steps:
Further tuning of the Logistic Regression model.
Experiment with advanced models such as Random Forests or Gradient Boosting to improve performance.
Explore methods to address class imbalance, such as using SMOTE or adjusting the classification threshold.
Dependencies
This project requires the following Python libraries:

Pandas: For data manipulation and analysis.
Scikit-learn: For machine learning model development and evaluation.
Matplotlib: For plotting visualizations (if applicable).

Conclusion
This project demonstrates how machine learning can be used to predict loan risk status based on financial features. The Logistic Regression model provides a foundation for risk assessment, and additional work is needed to improve performance, especially in terms of recall for high-risk loans.

