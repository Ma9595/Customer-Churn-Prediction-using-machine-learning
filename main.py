import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load the Datasets
# -----------------------------
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

print("Dataset Shapes (raw):")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# -----------------------------
# 2. Preprocessing Function
# -----------------------------
def preprocess_df(df):
    # Drop columns that are either IDs or directly related to churn (leakage features)
    drop_cols = []
    if 'customerID' in df.columns:
        drop_cols.append('customerID')
    for col in ['Churn Category', 'Churn Reason', 'Churn Score', 'Customer Status']:
        if col in df.columns:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Convert 'TotalCharges' to numeric if present (non-convertible become NaN)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical variables using LabelEncoder
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df


# Apply preprocessing to all datasets
train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)

print("\nAfter Preprocessing:")
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# -----------------------------
# 3. Explore Target Distribution
# -----------------------------
print("\nTrain Target Distribution:")
print(train_df['Churn'].value_counts())
print("\nValidation Target Distribution:")
print(val_df['Churn'].value_counts())
print("\nTest Target Distribution:")
print(test_df['Churn'].value_counts())

# -----------------------------
# 4. Split Features and Target for Training and Validation
# -----------------------------
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']

X_val = val_df.drop('Churn', axis=1)
y_val = val_df['Churn']

# -----------------------------
# 5. Model Training
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation on Validation Set
# -----------------------------
y_val_pred = model.predict(X_val)
acc_val = accuracy_score(y_val, y_val_pred)
conf_mat_val = confusion_matrix(y_val, y_val_pred)
class_report_val = classification_report(y_val, y_val_pred)

print("\nValidation Accuracy:", acc_val)
print("Validation Confusion Matrix:\n", conf_mat_val)
print("Validation Classification Report:\n", class_report_val)

# -----------------------------
# 7. Evaluation on Test Set (if target exists)
# -----------------------------
if 'Churn' in test_df.columns:
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']
    y_test_pred = model.predict(X_test)

    acc_test = accuracy_score(y_test, y_test_pred)
    conf_mat_test = confusion_matrix(y_test, y_test_pred)
    class_report_test = classification_report(y_test, y_test_pred)

    print("\nTest Accuracy:", acc_test)
    print("Test Confusion Matrix:\n", conf_mat_test)
    print("Test Classification Report:\n", class_report_test)
else:
    X_test = test_df
    y_test_pred = model.predict(X_test)
    print("\nTest Predictions (target column not found):\n", y_test_pred)

# -----------------------------
# 8. Feature Importance Visualization
# -----------------------------
importances = model.feature_importances_
feat_names = X_train.columns
feat_imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

print("\nFeature Importances DataFrame:")
print(feat_imp_df)
