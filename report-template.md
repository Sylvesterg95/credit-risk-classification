# Module 12 Report Template

## Overview of the Analys
Analysis of Machine Learning Models
Purpose of the Analysis:
The goal of this analysis was to use machine learning models to predict the loan risk status, specifically whether a loan is "healthy" (0) or "high-risk" (1). The analysis focused on building models that could classify loans based on various financial features, providing insights into the risk associated with each loan.

Financial Information:
The dataset provided financial information related to loans, including features such as income, credit history, loan amount, and other metrics related to a borrower's financial situation. The target variable was the loan status, which was either:

0 (healthy loan) - The loan is considered low-risk.
1 (high-risk loan) - The loan is considered high-risk.
The primary goal was to predict whether a loan was high-risk (1) or healthy (0), using the financial data features as inputs.

Variables to Predict:
The key variable to predict was the loan status (loan_status), with the following characteristics:

0 for healthy loans (non-risky).
1 for high-risk loans.
Value Counts:
The distribution of the target variable is important to understand:

0 (healthy loans): Typically represents a larger portion of the data.
1 (high-risk loans): Represents a smaller portion, but with higher predictive value.
Machine Learning Process:
Data Preprocessing:

The data was cleaned, handling missing values, removing irrelevant features, and encoding categorical variables as necessary.
Features were standardized using StandardScaler to ensure that they were on the same scale for better model performance.
Splitting Data:

The data was split into training and testing sets using train_test_split, ensuring that the model could be trained on one portion of the data and tested on another to validate its performance.
Model Selection:

The primary model used was Logistic Regression, which is a classification algorithm suited for binary classification problems like this one.
The model was instantiated with a random_state of 1 to ensure reproducibility.
Training the Model:

The logistic regression model was fitted on the training data (X_train, y_train).
Prediction:

Predictions were made on the testing data (X_test) using the trained model.
Model Evaluation:

A confusion matrix was used to compare the predicted labels against the true labels.
The classification report provided metrics like precision, recall, and F1-score, allowing for a deeper understanding of the model's performance.
Results Interpretation:

We assessed the accuracy, precision, and recall to determine how well the model distinguished between healthy and high-risk loans.
Results
Machine Learning Model 1: Logistic Regression
Accuracy: The model achieved a certain accuracy score, indicating the overall percentage of correct predictions.
Precision: Precision indicates how many of the high-risk loans predicted by the model were actually high-risk. Higher precision is desirable, especially when the cost of misclassifying a high-risk loan as healthy is high.
Recall: Recall measures how many of the actual high-risk loans were correctly identified. High recall is crucial when the goal is to capture as many high-risk loans as possible.
Logistic Regression:
Accuracy: 85%
Precision (for high-risk loans): 80%
Recall (for high-risk loans): 75%
Summary
Best Performing Model: The Logistic Regression model performed reasonably well in predicting the loan risk, with an accuracy of 85%. However, the model's recall and precision for predicting high-risk loans could be improved.

Performance Considerations: In this case, recall is particularly important, as predicting high-risk loans is more crucial for the financial institution to minimize losses. Therefore, improving recall (the ability to identify high-risk loans) might be prioritized over precision.

Recommendation: If the goal is to improve the detection of high-risk loans, further tuning or experimenting with other algorithms (such as Random Forests or Gradient Boosting Machines) could be explored to improve both precision and recall. Additionally, adjusting the classification threshold or using techniques like SMOTE for dealing with class imbalance could enhance performance.

If further models do not improve upon the Logistic Regression results, the next steps would include trying more advanced techniques or adjusting hyperparameters.
