# 🏦 Credit Risk Modeling — Loan Default Prediction

This project aims to predict the probability of a client defaulting on a loan (`loan_status = 1`) using machine learning techniques, specifically **Logistic Regression** with L1 and L2 regularization.

## 📂 Dataset
The project uses the **[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)** available on Kaggle. This dataset contains information on borrowers (age, income, home ownership status) and loan characteristics (amount, interest rate, grade).

---

## 🚀 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Oumaimazk/Credit-Risk-Modeling-Loan-Default-Prediction.git
   cd Credit-Risk-Modeling-Loan-Default-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**:
   Download the `credit_risk_dataset.csv` file and place it in the project root folder.

4. **Launch the analysis**:
   ```bash
   jupyter notebook credit_risk_model.ipynb
   ```

---

## 🗂️ Project Structure
```
credit-risk-modeling/
│
├── credit_risk_model.ipynb   ← Main Notebook (Analysis & Modeling)
├── README.md                 ← Project Description
├── requirements.txt          ← Python Dependencies
├── .gitignore                ← Files ignored by Git
└── credit_risk_dataset.csv   ← Dataset (add locally)
```

---

## 📝 Project Analysis

### 1. Exploratory Data Analysis (EDA)
The project begins with a thorough analysis of the variables:
- **Numerical Variables**: Age, Income, Loan Amount, Interest Rate.
- **Categorical Variables**: Home Ownership (RENT, MORTGAGE, OWN), Loan Intent (EDUCATION, MEDICAL, etc.).
- **Missing Values Treatment**: Median imputation for employment length and interest rate.

### 2. Preprocessing
- **Encoding**: Transformation of textual variables into numerical ones via `LabelEncoder`.
- **Scaling**: Normalization of data with `StandardScaler` to ensure logistic regression convergence.

### 3. Modeling
Using **Logistic Regression** allows for clear interpretation of default probabilities. We compare models without regularization, with **Lasso (L1)** and **Ridge (L2)** to prevent overfitting.

---

## 🧮 Mathematical Foundations

The project integrates the following fundamental mathematical concepts:

### A. The Sigmoid Function
To transform the linear score $z = \mathbf{w}^T \mathbf{x} + b$ into a probability between 0 and 1:
$$P(Y=1 | \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### B. Maximum Likelihood Estimation (MLE)
The model seeks to maximize the probability of observing the real data. The minimized loss function is the **Binary Cross-Entropy**:
$$J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$$

### C. Regularization (L1 & L2)
To limit model complexity:
- **L1 (Lasso)**: Adds $\lambda \sum |w_j|$, promoting sparsity (some weights become zero).
- **L2 (Ridge)**: Adds $\lambda \sum w_j^2$, penalizing large coefficients for more stability.

### D. Evaluation: ROC Curve and AUC
We use the **Area Under the Curve (AUC)** to measure the model's ability to distinguish between defaulting and healthy clients, independent of the chosen classification threshold.

---

## 📊 Results
Model performance is evaluated via:
- **Confusion Matrix** (True Positives, False Negatives).
- **Classification Report** (Precision, Recall, F1-Score).
- **Covariance Matrix stability**.

---
*Developed as part of a Credit Risk Data Science analysis.*
