# Financial Fraud Detection Using Machine Learning

An end-to-end machine learning project focused on identifying fraudulent financial transactions through exploratory data analysis, feature engineering, class imbalance handling, and comparative model evaluation.

## Overview

Financial fraud is a major challenge for banking institutions and digital payment platforms. Detecting fraudulent transactions accurately is essential for minimizing financial losses, protecting customers, and improving transaction security.

This project develops a binary classification pipeline to identify potentially fraudulent financial transactions using historical transaction data. It explores multiple machine learning algorithms, addresses extreme class imbalance, and evaluates model performance using metrics relevant to fraud detection.

The primary objective is to distinguish legitimate transactions from fraudulent ones while balancing fraud detection capability and false-positive rates.

## Project Highlights

- Analysis of approximately **6.36 million financial transactions**.
- Exploratory Data Analysis (EDA) to identify transaction patterns and anomalies.
- Feature engineering and statistical feature selection.
- Handling extreme class imbalance using SMOTE and class-weighting techniques.
- Comparative evaluation of four machine learning algorithms.
- Hyperparameter optimization using GridSearchCV.
- Model evaluation using precision, recall, F1-score, ROC-AUC, and Precision-Recall curves.
- Business-oriented recommendations for fraud monitoring and detection systems.

## Dataset Overview

The project works with a large financial transaction dataset containing approximately 6.36 million records and 11 initial features.

| Attribute | Description |
|---|---|
| Total Transactions | 6,362,620 |
| Initial Features | 11 |
| Fraudulent Transactions | 8,213 |
| Target Variable | `isFraud` |
| Problem Type | Binary Classification |
| Fraud Percentage | Approximately 0.13% |

**Target Variable:**

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The dataset presents a significant class imbalance challenge, making fraud detection considerably more difficult than conventional classification problems.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Imbalanced Learning | Imbalanced-learn (SMOTE) |
| Gradient Boosting | XGBoost |
| Model Optimization | GridSearchCV |
| Statistical Analysis | Variance Inflation Factor (VIF), Mutual Information |

## Machine Learning Workflow

The project follows a structured machine learning workflow, covering the complete process from raw data exploration to model evaluation.

### 1. Data Loading and Exploratory Data Analysis

- Loaded and inspected the financial transaction dataset.
- Examined data types, dimensions, missing values, and memory consumption.
- Analyzed the distribution of legitimate and fraudulent transactions.
- Explored relationships between numerical features.
- Visualized correlations and transaction patterns.

### 2. Data Cleaning and Preprocessing

Prepared the dataset for machine learning by:

- Identifying and handling missing values.
- Removing features with excessive missing data.
- Applying median and mode imputation where appropriate.
- Analyzing numerical outliers.
- Checking multicollinearity using Variance Inflation Factor (VIF).
- Applying appropriate transformations and scaling.

### 3. Feature Engineering and Selection

Created and evaluated relevant features to improve the predictive capabilities of the models.

Techniques included:

- Logarithmic transformations of numerical variables.
- Feature scaling and normalization.
- Mutual Information for identifying informative features.
- Random Forest feature importance for ranking predictive variables.
- Statistical analysis to understand relationships between transaction attributes.

### 4. Handling Class Imbalance

One of the most important challenges was the extremely low proportion of fraudulent transactions.

With only approximately 0.13% fraudulent records, conventional classification models could become biased toward predicting legitimate transactions.

To address this challenge, the project explored:

- **SMOTE:** Generates synthetic minority-class samples to improve representation during model training.
- **Class Weighting:** Assigns greater importance to fraudulent transactions during model optimization.
- **Scale Positive Weight:** Applies class imbalance adjustments for XGBoost.

Resampling techniques were applied to training data to avoid introducing synthetic samples into validation and test datasets.

### 5. Model Development

Four machine learning algorithms were explored and compared:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Establishes a baseline classification model. |
| Random Forest | Captures nonlinear relationships and feature interactions. |
| Gradient Boosting | Builds sequential models to improve predictive performance. |
| XGBoost | Provides an optimized gradient boosting approach for structured data. |

The models were evaluated using a consistent validation methodology to understand their ability to identify fraudulent transactions.

### 6. Hyperparameter Optimization

GridSearchCV with 5-fold cross-validation was incorporated to identify suitable model hyperparameters.

The optimization process focused on improving classification performance while considering the challenges associated with highly imbalanced data.

### 7. Model Evaluation

Accuracy alone is not a reliable evaluation metric for this problem because a model could achieve high accuracy simply by predicting almost every transaction as legitimate.

Therefore, the evaluation focuses on:

| Metric | Importance |
|---|---|
| Precision | Measures how many transactions flagged as fraudulent are actually fraudulent. |
| Recall | Measures how many actual fraudulent transactions are successfully identified. |
| F1-Score | Balances precision and recall. |
| ROC-AUC | Measures the model's ability to distinguish between the two classes. |
| Precision-Recall Curve | Helps assess performance under extreme class imbalance. |
| Confusion Matrix | Provides insight into true positives, false positives, true negatives, and false negatives. |

These metrics provide a more meaningful understanding of fraud detection capabilities and the trade-off between missed fraud and unnecessary alerts.

## Business Applications

The techniques explored in this project can support practical financial fraud monitoring systems.

Potential applications include:

- **Real-Time Transaction Monitoring:** Integrating a trained model into transaction processing systems to assess suspicious activity.
- **Risk-Based Alerting:** Assigning risk scores to transactions and triggering alerts based on configurable thresholds.
- **Fraud Investigation:** Helping fraud analysts prioritize potentially suspicious transactions.
- **Performance Monitoring:** Tracking precision, recall, false-positive rates, and changes in transaction patterns.
- **Model Retraining:** Periodically updating models using newer transaction data to adapt to evolving fraud patterns.

These are proposed applications rather than currently implemented production features.

## Key Challenges and Learnings

Throughout this project, several important machine learning challenges were explored:

1. Working with a large dataset containing more than six million records.
2. Understanding the limitations of accuracy in highly imbalanced classification problems.
3. Selecting meaningful features using statistical and model-based techniques.
4. Applying SMOTE and class-weighting approaches to improve minority-class learning.
5. Comparing different classification algorithms using appropriate evaluation metrics.
6. Understanding how model predictions can translate into practical fraud prevention strategies.

## Repository Structure

```text
Fraud-Detection/
│
├── Fraud_Detection.ipynb    # Complete ML pipeline
│
└── README.md                # Project documentation
```

## Getting Started

### Prerequisites

Make sure Python 3.9 or later is installed.

### 1. Clone the Repository

```bash
git clone https://github.com/Kavya-kakkar/Fraud-Detection.git
cd Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Fraud_Detection.ipynb` to explore the complete implementation.

**Note:** The original dataset is not included in this repository. Place the dataset in the appropriate location expected by the notebook before executing the cells.

## Future Improvements

Potential directions for extending this project include:

- Experimenting with LightGBM and CatBoost.
- Evaluating anomaly detection techniques such as Isolation Forest.
- Exploring cost-sensitive learning for fraud classification.
- Developing an interactive fraud monitoring dashboard.
- Building a REST API using FastAPI for model inference.
- Implementing model explainability using SHAP.
- Monitoring concept drift and implementing automated model retraining.

## Conclusion

This project demonstrates an end-to-end approach to financial fraud detection using machine learning. It combines exploratory data analysis, preprocessing, feature engineering, imbalance handling, model development, and performance evaluation into a structured workflow.

Beyond building classification models, the project explores how machine learning techniques can contribute to practical financial risk management and transaction security.

---

**Author:** [Kavya Kakkar](https://github.com/Kavya-kakkar)

**GitHub Repository:** [Financial Fraud Detection](https://github.com/Kavya-kakkar/Fraud-Detection)

**Implementation:** Python | Machine Learning | Data Science
