# Credit Risk Scorecard

**Live Demo:** [Launch App](https://credit-portfolio-risk-analytics-hhrbkmnkqvtgwfrh5u5rtp.streamlit.app/)
## Overview
End-to-end credit risk scoring system built on Home Credit Default Risk dataset (307,511 loans).

## Features
- EDA with null analysis, feature distributions, correlation heatmap
- Information Value (IV/WoE) based feature selection — 24 features selected from 122
- Logistic Regression baseline (AUC: 0.739)
- XGBoost model (AUC: 0.751, KS: 0.37, Gini: 0.50)
- SHAP explainability — EXT_SOURCE scores identified as top predictors
- Claude AI layer for plain-English risk assessment
- Streamlit dashboard for real-time applicant scoring

## Tech Stack
Python, XGBoost, SHAP, Scikit-learn, Streamlit, Anthropic Claude API

## Dataset
Home Credit Default Risk (Kaggle) — 8.1% default rate, class imbalance handled via scale_pos_weight
