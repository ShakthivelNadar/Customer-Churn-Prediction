# Customer-Churn-Prediction
Customer Churn Prediction
Project Overview
This project focuses on predicting customer churn for a telecommunications company. Customer churn, the loss of customers, is a critical issue for businesses. By accurately predicting which customers are likely to churn, companies can implement targeted retention strategies to minimize losses.

Dataset
The dataset used in this project is the Telco Customer Churn dataset, a publicly available dataset commonly used for churn prediction tasks. It contains customer demographics, services subscribed to, account information, and churn status.

Key Features:

gender: Customer's gender (Female, Male)
SeniorCitizen: Whether the customer is a senior citizen (1, 0)
Partner: Whether the customer has a partner (Yes, No)
Dependents: Whether the customer has dependents (Yes, No)
tenure: Number of months the customer has stayed with the company
PhoneService: Whether the customer has phone service (Yes, No)
MultipleLines: Whether the customer has multiple lines (Yes, No phone service, No)
InternetService: Customer's internet service provider (DSL, Fiber optic, No)
OnlineSecurity: Whether the customer has online security (Yes, No internet service, No)
OnlineBackup: Whether the customer has online backup (Yes, No internet service, No)
DeviceProtection: Whether the customer has device protection (Yes, No internet service, No)
TechSupport: Whether the customer has tech support (Yes, No internet service, No)
StreamingTV: Whether the customer has streaming TV (Yes, No internet service, No)
StreamingMovies: Whether the customer has streaming movies (Yes, No internet service, No)
Contract: The contract term (Month-to-month, One year, Two year)
PaperlessBilling: Whether the customer has paperless billing (Yes, No)
PaymentMethod: The payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
MonthlyCharges: The amount charged to the customer monthly
TotalCharges: The total amount charged to the customer
Target Variable:

Churn: Whether the customer churned or not (Yes, No)
Methodology
1. Data Preprocessing
Feature Dropping: The customerID column was dropped as it is a unique identifier and not relevant for modeling.
Data Type Conversion: The TotalCharges column, initially an object type, was converted to a numeric type. Missing values introduced by this conversion were filled with the median of the column.
Target Encoding: The Churn column was converted from categorical ('Yes', 'No') to numerical (1, 0).
Categorical Encoding: All categorical features were one-hot encoded using pd.get_dummies with drop_first=True to avoid multicollinearity.
Feature Scaling: Numerical features were scaled using StandardScaler to normalize their ranges, which helps Gradient Boosting models perform better.
Data Splitting: The dataset was split into training (80%) and testing (20%) sets using train_test_split with stratify=y to maintain the same proportion of churned and non-churned customers in both sets.
2. Model Training
A GradientBoostingClassifier was chosen for its strong performance in classification tasks.
Hyperparameter Tuning: GridSearchCV was employed to find the optimal hyperparameters for the Gradient Boosting model. The following hyperparameters were tuned:
n_estimators: [200, 300]
learning_rate: [0.05, 0.1]
max_depth: [3, 4]
The model was trained on the scaled training data.
3. Model Evaluation
The best model identified by GridSearchCV was used to make predictions on the test set.
Evaluation metrics included:
Accuracy: Overall correctness of the model's predictions.
Classification Report: Provides precision, recall, and f1-score for each class.
Confusion Matrix: Shows the counts of true positive, true negative, false positive, and false negative predictions.
Results
The model achieved an accuracy of approximately 80.27% on the test set. The classification report and confusion matrix provide a more detailed view of the model's performance, particularly regarding its ability to identify churned customers.

Key Feature Importances: An analysis of feature importances revealed that tenure, InternetService_Fiber optic, PaymentMethod_Electronic check, Contract_Two year, TotalCharges, Contract_One year, and MonthlyCharges are among the most significant predictors of customer churn.

Prediction on New Data
The trained model can be used to predict churn for new, unseen customer data. The new data undergoes the same preprocessing steps (dropping customerID, one-hot encoding, and scaling) before being fed to the model for prediction.

How to Run
Clone the repository:
git clone <repository_url>
cd customer-churn-prediction
Open in Google Colab: Upload the .ipynb notebook to Google Colab.
Run all cells: Execute all cells in the notebook sequentially.
Data Path: Ensure the WA_Fn-UseC_-Telco-Customer-Churn.csv file is accessible at /content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv or update the path in the notebook.
Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
