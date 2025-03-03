import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
import pickle

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Regression models
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

# App title and sidebar
st.set_page_config(page_title="AutoMLX", page_icon="🤖", layout="wide")

# Your name for attribution
YOUR_NAME = "Varshitha"
PROJECT_NAME = "AutoMLX"

# Sidebar
authentication_status = 1
if authentication_status:
    with st.sidebar:
        st.image("https://cdn.pixabay.com/photo/2021/08/04/13/06/software-developer-6521720_1280.jpg", use_column_width=True)
        st.title(PROJECT_NAME)
        st.info(f"Created by {YOUR_NAME}")
        choice = st.radio("Navigation", ["Home", "Upload", "EDA", "Data Cleaning", "Modelling", "Download"])
        st.info("This application automates your machine learning pipeline, making data analysis and model building effortless.")

# Home Page
if choice == "Home":
    st.title(f"Welcome to {PROJECT_NAME}: Your Automated Machine Learning Solution!")
    st.write(f"""
        Hi, I'm {YOUR_NAME}, and I'm excited to introduce you to **{PROJECT_NAME}**, a powerful tool designed to simplify and automate your machine learning workflow. Whether you're a beginner or an experienced data scientist, {PROJECT_NAME} helps you build, analyze, and deploy machine learning models with ease.

        ### **What {PROJECT_NAME} Offers**
        - **Automated Data Cleaning**: Handle missing values, outliers, and inconsistencies effortlessly.
        - **Smart Preprocessing**: Automatically encode, scale, and transform your data for optimal results.
        - **Exploratory Data Analysis (EDA)**: Generate insightful visualizations to understand your data better.
        - **Model Selection & Training**: Automatically select and train the best model for your task.
        - **Visualization & Reporting**: Get clear, interactive visualizations of model performance and data insights.

        ### **Why Choose {PROJECT_NAME}?**
        - **Save Time**: Automate repetitive tasks and focus on interpreting results.
        - **No Expertise Required**: Perfect for users with limited machine learning knowledge.
        - **Scalable**: Handles diverse datasets and adapts to various machine learning tasks.

        ### **How It Works**
        1. **Upload Your Dataset**: Provide your dataset in CSV or another supported format.
        2. **Let {PROJECT_NAME} Work**: The pipeline automatically cleans, preprocesses, and analyzes your data.
        3. **Train Models**: {PROJECT_NAME} selects and trains the best model for your task.
        4. **Explore Results**: Visualize performance metrics, insights, and predictions.

        ### **Future Plans**
        In the future, we plan to incorporate **Explainable AI (XAI)** to provide increased transparency and understanding of the model's decision-making process.

        Thank you for choosing **{PROJECT_NAME}** for your data analysis needs. I hope you find it helpful and user-friendly!
    """)

# Upload Dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset (CSV format)", type=["csv"])
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

# Exploratory Data Analysis (EDA)
if choice == "EDA":
    st.title("Exploratory Data Analysis")
    try:
        df = pd.read_csv('dataset.csv', index_col=None)
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    except Exception as e:
        st.error("Please upload a dataset first or ensure the dataset is valid.")

# Data Cleaning
if choice == "Data Cleaning":
    st.title("Data Cleaning")
    try:
        df = pd.read_csv('dataset.csv', index_col=None)
        threshold_cols = int(df.shape[0] * 0.5)
        df.dropna(axis=1, thresh=threshold_cols, inplace=True)

        for col in df.columns:
            if df[col].dtype not in ['int64', 'float64']:
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df.drop(col, axis=1, inplace=True)
        df.interpolate(method='linear', inplace=True)
        st.success("Data cleaned successfully!")
        st.dataframe(df)
        df.to_csv('dataset.csv', index=None)
    except Exception as e:
        st.error("Please upload a dataset first or ensure the dataset is valid.")

# Modelling
if choice == "Modelling":
    st.title("Model Training")
    try:
        df = pd.read_csv('dataset.csv', index_col=None)
        classoreg = st.radio("Choose the type of problem", ["Regression", "Classification"])
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        X = df.drop(chosen_target, axis=1)
        y = df[chosen_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        if classoreg == "Regression":
            if st.button('Run Modelling'):
                models = [
                    ('Multiple Linear Regression', LinearRegression()),
                    ('Polynomial Regression', PolynomialFeatures()),
                    ('Robust Regression - RANSAC', RANSACRegressor()),
                    ('Decision Tree', DecisionTreeRegressor()),
                    ('Random Forest', RandomForestRegressor()),
                    ('Gaussian Process Regression', GaussianProcessRegressor()),
                    ('Support Vector Regression', SVR())
                ]
                param_grid = {
                    'Multiple Linear Regression': {},
                    'Polynomial Regression': {'degree': [2, 3, 4]},
                    'Robust Regression - RANSAC': {'max_trials': [100, 200, 500], 'min_samples': [10, 20, 30]},
                    'Decision Tree': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
                    'Random Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [5, 10, 20, None]},
                    'Gaussian Process Regression': {'kernel': [None, 'RBF']},
                    'Support Vector Regression': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                }
                best_model_obj, best_score = train_models(models, param_grid, X, y)
                st.success(f"The best model is {best_model_obj} with a score of {best_score}")
                test_score = best_model_obj.score(X_test, y_test)
                st.success(f"The test score for the best model is {test_score}")
                pickle.dump(best_model_obj, open('best_model.pkl', 'wb'))
        else:
            if st.button('Run Modelling'):
                models = [
                    ('Logistic Regression', LogisticRegression()),
                    ('Random Forest', RandomForestClassifier()),
                    ('K-Nearest Neighbors', KNeighborsClassifier()),
                    ('Support Vector Machine', SVC()),
                    ('Gaussian Naive Bayes', GaussianNB()),
                    ('XGBoost', XGBClassifier())
                ]
                param_grid = {
                    'Logistic Regression': {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']},
                    'Random Forest': {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [5, 10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                    'K-Nearest Neighbors': {'n_neighbors': [1, 3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
                    'Support Vector Machine': {'C': np.logspace(-3, 3, 7), 'kernel': ['linear', 'rbf'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
                    'Gaussian Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]},
                    'XGBoost': {'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 5, 7, 10], 'n_estimators': [100, 200, 500], 'booster': ['gbtree', 'dart'], 'subsample': [0.8, 0.9, 1], 'colsample_bytree': [0.8, 0.9, 1]}
                }
                best_model_obj, best_score = train_models(models, param_grid, X, y)
                st.success(f"The best model is {best_model_obj} with a score of {best_score}")
                test_score = best_model_obj.score(X_test, y_test)
                st.success(f"The test score for the best model is {test_score}")
                pickle.dump(best_model_obj, open('best_model.pkl', 'wb'))
    except Exception as e:
        st.error("Please upload a dataset first or ensure the dataset is valid.")

# Download Model
if choice == "Download":
    st.title("Download Your Model")
    try:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    except:
        st.error("Please complete the Modelling section first.")

# Helper function for model training
def train_models(models, param_grid, X, y):
    best_model_obj = None
    best_score = -np.inf
    for model_name, model in models:
        grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid[model_name], cv=5, n_jobs=3)
        grid.fit(X, y)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model_obj = grid.best_estimator_
    return best_model_obj, best_score
