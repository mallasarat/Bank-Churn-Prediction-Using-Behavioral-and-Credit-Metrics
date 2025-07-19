# 💼 Customer Retention Intelligence: Predicting Bank Churn Using Python & Power BI

## 📌 Project Overview
This project uses predictive analytics to identify churn risks in a banking dataset. By analyzing customer demographics, behavior, and credit utilization, it empowers business teams to take action on high-risk segments through insights and strategic outreach.

---

## 🔍 Objective
To build a full-stack analytics pipeline that predicts customer churn and translates model insights into business recommendations—leveraging Python for modeling and Power BI for dashboarding.

---

## 🛠️ Tools & Technologies
- **Languages & Libraries**: Python, pandas, seaborn, scikit-learn, XGBoost  
- **Visualization**: Power BI  
- **Dataset**: [Bank Churners Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (~10,000 records)  

---

## 📊 Methodology

### 1️⃣ Data Exploration (EDA)
- Analyzed churn distribution (`Attrited` vs `Existing`)
- Found strong signals in:
  - Low transaction count (`Total_Trans_Ct`)
  - High credit utilization (`Avg_Utilization_Ratio`)
- Explored patterns across age, income, education, and card types

### 2️⃣ Feature Engineering
- Encoded categorical variables and scaled numeric features
- Created age groups and risk segments for dashboard filtering

### 3️⃣ Predictive Modeling
- Built and tuned **XGBoost classifier**
- Achieved:
  - **Accuracy**: ~86%
  - **ROC-AUC**: 0.91
- Evaluated using precision, recall, and F1-score

### 4️⃣ Business Dashboard (Power BI)
- KPIs: Total Customers, Churn Rate %, Avg Utilization
- Churn segmentation charts by income, education, card category
- Behavioral visuals: utilization vs transactions
- Slicers: gender, age group, marital status

---

## 📈 Key Business Insights
- Customers with **low transaction volume + high utilization** are more likely to churn  
- Middle-income, mid-age groups (30–45) showed unexpected churn spikes  
- Recommended strategies:
  - Personalized outreach for high-risk cohorts  
  - Engagement incentives for low-activity, high-credit users  

---

## 🧠 Key Takeaways
- Built a robust predictive solution from EDA to dashboard  
- Demonstrated fluency in both **technical modeling** and **strategic storytelling**  
- Created an interactive dashboard for dynamic exploration and stakeholder communication

---

## 📂 Files Included
- `BankChurners.csv`: Original dataset  
- `BankChurners_Processed.csv`: Cleaned, engineered dataset for dashboarding  
- `churn_model_pipeline.py`: Full Python script for EDA, modeling, and export  
- Power BI `.pbix` file *(optional)*

---

## 📣 Author
**Sarat** – Data Analyst passionate about turning models into strategic business insights.  
🔗 [LinkedIn](http://www.linkedin.com/in/mallasarat) | 📫 [Email](mallasarat200@gmail.com)

---
