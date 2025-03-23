# Customer-Churn-Prediction-using-machine-learning
## 📊 Customer Churn Prediction

**Summary:**  
This project aims to predict whether a customer will churn using a Random Forest classifier. After careful data cleaning and removal of leakage features, the model achieved:

-  Validation Accuracy: **96.65%**
-  Test Accuracy: **94.24%**
-  Top Feature: **Satisfaction Score** (40% importance)

**Dataset:**  
A telecom dataset containing customer demographics, service usage, and satisfaction scores.

**Key Steps:**
- Preprocessing with Label Encoding and NaN handling
- Leakage removal (dropped `Churn Category`, `Churn Score`, etc.)
- Feature importance analysis
- Evaluation using accuracy, precision, recall, F1-score

**Top 5 Features:**
1. Satisfaction Score  
2. Tenure in Months  
3. Total Charges  
4. Contract  
5. Offer  

**Tools Used:**  
Python, Pandas, Scikit-learn, Matplotlib, Seaborn
