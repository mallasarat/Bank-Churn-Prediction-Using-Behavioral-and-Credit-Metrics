# 📦 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

# 🔹 Load Dataset
df = pd.read_csv("BankChurners.csv")

# 🧾 Data Overview
print(df.info())
print(df.describe())
print("Churn Distribution:\n", df['Attrition_Flag'].value_counts(normalize=True))
print("Missing Values:\n", df.isnull().sum())

# 🧹 Data Cleaning
df.drop_duplicates(inplace=True)

# 🎨 Visualizations
sns.set(style='whitegrid')

# 1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition_Flag', data=df)
plt.title('Churn Distribution')
plt.tight_layout()
plt.show()

# 2. Customer Age vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition_Flag', y='Customer_Age', data=df)
plt.title('Age vs Churn')
plt.tight_layout()
plt.show()

# 3. Transaction Count Histogram
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Total_Trans_Ct', hue='Attrition_Flag', kde=True, multiple='stack')
plt.title('Transaction Count Distribution')
plt.tight_layout()
plt.show()

# 4. Utilization Ratio vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition_Flag', y='Avg_Utilization_Ratio', data=df)
plt.title('Utilization Ratio vs Churn')
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(12,8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.tight_layout()
plt.show()

# 🎯 Feature Engineering

# 1. Target Encoding
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# 2. One-Hot Encoding for Categorical Features
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Scaling Numeric Features
scaler = StandardScaler()
num_cols = ['Customer_Age', 'Credit_Limit', 'Avg_Utilization_Ratio', 'Total_Trans_Ct']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 🤖 Model Training

# Split data
X = df.drop(['Attrition_Flag'], axis=1)
y = df['Attrition_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 🧪 Evaluation
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(" ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# 📦 Imports
df.to_csv("BankChurners_Processed.csv", index=False)