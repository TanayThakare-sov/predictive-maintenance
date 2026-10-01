# 🔧 Intelligent Predictive Maintenance & Failure Risk Analysis

An end-to-end Machine Learning and Deep Learning project for predicting machine failure risk from industrial operating conditions.

The system analyzes machine parameters such as temperature, rotational speed, torque, and tool wear to estimate the probability of machine failure and classify the machine into Low, Medium, or High risk.

---

## 📌 Project Overview

Unexpected machine failures can cause production downtime, emergency maintenance, and additional operational costs.

This project uses historical machine operating data to build predictive models that identify machines with an elevated risk of failure.

The project compares multiple Machine Learning and Deep Learning approaches and uses explainability techniques to understand why the model makes its predictions.

The final trained XGBoost model is deployed through a Streamlit web application.

---

## 🎯 Objectives

- Predict machine failure using machine operating conditions
- Handle highly imbalanced failure data
- Compare multiple ML and DL models
- Analyze false positives and false negatives
- Identify important features affecting predictions
- Provide failure probability instead of only a binary prediction
- Classify machines into Low, Medium, and High risk
- Deploy the trained model using Streamlit

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

### Dataset size

- Rows: **10,000**
- Original columns: **14**
- Machine failures: **339**
- Non-failures: **9,661**

### Machine failure distribution

| Machine Failure | Count | Percentage |
|---|---:|---:|
| No Failure | 9,661 | 96.61% |
| Failure | 339 | 3.39% |

The dataset is highly imbalanced, so class imbalance was considered during model development.

---

## 🔎 Features Used

The final predictive models use:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The categorical `Type` feature is converted using one-hot encoding.

The resulting model features are:

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Type_L
Type_M
