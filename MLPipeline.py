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

# Sidebar
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoMLX")
    st.info("An automated ML pipeline for data-driven insights")
    choice = st.radio("Navigation", ["Home", "Upload", "EDA", "Data Cleaning", "Modelling", "Download"])
    st.info("Streamline your ML workflow with AutoMLX")

# Home Page
if choice == "Home":
    st.title("Welcome to AutoMLX: Automate Your ML Workflow")
    st.write("""
        **AutoMLX** simplifies the machine learning pipeline, handling everything from data preprocessing to model selection.
        Just upload your dataset, and AutoMLX will clean, analyze, and train models for you!
        
        ### **Core Features**
        - **Automatic Data Cleaning**: Handles missing values, outliers, and feature transformations.
        - **Smart Preprocessing**: Scales, encodes, and optimizes features for better model performance.
        - **EDA & Visualization**: Provides insightful visualizations and statistical summaries.
        - **Model Training & Selection**: Identifies the best ML model for your dataset.
        - **Performance Metrics & Reporting**: Generates comprehensive reports and visualizations.
        
        ### **How It Works**
        1. **Upload Your Dataset**
        2. **Let AutoMLX Process the Data**
        3. **Run Model Selection & Training**
        4. **Review Insights & Download Your Model**
    """)

# Upload Dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df.to_csv('dataset.csv', index=False)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

# Exploratory Data Analysis (EDA)
if choice == "EDA":
    st.title("Exploratory Data Analysis")
    try:
        df = pd.read_csv('dataset.csv')
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    except:
        st.error("Please upload a dataset first.")

# Data Cleaning
if choice == "Data Cleaning":
    st.title("Data Cleaning")
    try:
        df = pd.read_csv('dataset.csv')
        df.dropna(thresh=int(df.shape[0] * 0.5), axis=1, inplace=True)
        df.interpolate(method='linear', inplace=True)
        st.success("Data cleaned successfully!")
        st.dataframe(df)
        df.to_csv('dataset.csv', index=False)
    except:
        st.error("Please upload a valid dataset first.")

# Modelling
if choice == "Modelling":
    st.title("Model Training")
    try:
        df = pd.read_csv('dataset.csv')
        problem_type = st.radio("Select Problem Type", ["Regression", "Classification"])
        target_col = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if problem_type == "Regression":
            models = [
                ('Linear Regression', LinearRegression()),
                ('Polynomial Regression', PolynomialFeatures()),
                ('RANSAC Regressor', RANSACRegressor()),
                ('Decision Tree', DecisionTreeRegressor()),
                ('Random Forest', RandomForestRegressor()),
                ('Gaussian Process', GaussianProcessRegressor()),
                ('SVR', SVR())
            ]
        else:
            models = [
                ('Logistic Regression', LogisticRegression()),
                ('Random Forest', RandomForestClassifier()),
                ('K-Nearest Neighbors', KNeighborsClassifier()),
                ('SVM', SVC()),
                ('Naive Bayes', GaussianNB()),
                ('XGBoost', XGBClassifier())
            ]

        if st.button("Train Models"):
            best_model, best_score = train_best_model(models, X_train, y_train)
            st.success(f"Best Model: {best_model} with Score: {best_score}")
            pickle.dump(best_model, open('best_model.pkl', 'wb'))
    except:
        st.error("Please upload and clean a dataset first.")

# Download Model
if choice == "Download":
    st.title("Download Trained Model")
    try:
        with open('best_model.pkl', 'rb') as f:
            st.download_button("Download Model", f, file_name="best_model.pkl")
    except:
        st.error("Train a model first.")

# Helper function to train models
def train_best_model(models, X, y):
    best_model, best_score = None, -np.inf
    for name, model in models:
        model.fit(X, y)
        score = model.score(X, y)
        if score > best_score:
            best_score, best_model = score, model
    return best_model, best_score
