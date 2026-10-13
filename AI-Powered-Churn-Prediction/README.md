<h1 align="center">🛡️ Customer Churn Prediction Dashboard</h1>
<p align="center">
  An intelligent, interactive dashboard built with <strong>Streamlit</strong> + <strong>Plotly</strong><br>
  for analyzing, visualizing, and predicting customer churn with real-time insights.
</p>

<p align="center">
  <span>👨‍💻 Author: Ashwini Vishal</span>
</p>

---

## 🚀 Overview

The **Churn Prediction Dashboard** is a machine learning-powered web app that identifies which customers are at risk of leaving a service. Built with real-world telecom data, this tool helps business teams proactively engage with users before churn happens.

It features:
- Interactive churn prediction
- Real-time risk recommendations
- Gender-wise and contract-wise visual insights
- Future scenario simulation
- CSV download support

---

## 🧩 Features

✅ **Predict churn probability per customer (0 = Active, 1 = Churned)**  
📈 **Visualize top churn drivers using feature importance**  
🔍 **Smart charts: Tenure, Monthly Charges, Gender Analysis**  
🔮 **Future Churn Simulator (change charges & tenure)**  
📥 **Download full customer dataset**  
🌑 **Sleek dark-mode dashboard UI**  
🧠 **Trained with XGBoost ML Model**

---

## 🛠 Tech Stack Used

This project combines modern ML tools with a rich frontend:

- **📊 Streamlit** – For interactive dashboard UI
- **📈 Plotly & Seaborn** – Visualizing customer trends
- **🐍 Python** – Core logic and modeling
- **🧮 Pandas & NumPy** – Data preprocessing & handling
- **🧠 Scikit-learn & XGBoost** – Model training and evaluation
- **📄 CSV Dataset** – Kaggle’s Telco Customer Churn Dataset

---

## 🧠 Machine Learning Model

We use [XGBoost](https://xgboost.readthedocs.io/en/stable/) as the main model:
- Trained on engineered features like `tenure`, `contract`, `monthly charges`, etc.
- High accuracy in predicting churn likelihood
- Feature importance extracted for business interpretation

![](Assets/churn5.png)

---

## 📸 Visual Insights

![](Assets/churn2.png)

![](Assets/churn3.png)

![](Assets/churn4.png)

![](Assets/churn6.png)

---

## 📁 Dataset Columns

We use the following key columns after cleaning and encoding:

```csv
gender, tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, Churn
```

## ▶️ Getting Started

Ready to launch the AI Churn Prediction Dashboard on your machine? Follow these simple steps from the project folder:

---

### 🔧 Setup

```powershell
py -m pip install -r requirements.txt
py -m streamlit run churn_prediction_dashboard.py
```

If your system uses `python` instead of `py`, replace `py` with `python`.
---

## 📥 Download Filtered Data 

🎯 **Purpose**: Empower users to understand, analyze, and extract customer behavior and churn risk — instantly!

The dashboard offers an intuitive and seamless way to:

- 🧠 **AI-Powered Risk Scores** per customer  
- 📊 **Top 10 Sample Customer Profiles** with:
  - Tenure
  - Charges
  - Senior Citizen status
  - Actual Churn status
  - Predicted Risk Level

Once explored, users can easily:

✅ **Download the full churn dataset as CSV**  
📤 **Use it for reporting, analysis, or business decisions**


---

## 🌟 Show Some Love

If you like this project, please consider:

⭐️ **Starring** the repo    

Let’s connect and grow together 🚀
