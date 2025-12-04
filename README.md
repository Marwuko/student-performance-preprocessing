# Student Performance Preprocessing & Modeling

This project explores how different **scikit-learn preprocessing techniques** affect the performance of machine learning models in predicting **SSC Overall GPA** for students.  
Using a dataset containing academic history, parental background, and school-related factors, the project evaluates multiple preprocessing pipelines and identifies the best strategy based on error metrics and R² score.

The project is fully implemented in Python using **pandas, scikit-learn, seaborn, matplotlib**, and **Jupyter/Colab notebooks**.

---

## 🎯 Project Objectives

- Perform complete **Exploratory Data Analysis (EDA)**  
- Apply and compare several preprocessing strategies:  
  - StandardScaler  
  - MinMaxScaler  
  - RobustScaler  
  - QuantileTransformer (uniform)  
  - PowerTransformer (Yeo–Johnson)  
  - No Scaling (baseline)  
- Build reproducible **Pipeline + ColumnTransformer** architectures  
- Train and evaluate machine learning models (RandomForestRegressor)  
- Determine which preprocessing method yields the best predictive performance  
- Visualize distributions, transformations, and model results  
- Produce a clean, well-structured machine learning workflow  

---

## 📁 Repository Structure
