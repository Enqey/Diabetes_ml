# -*- coding: utf-8 -*-
"""
Improved Diabetes Prediction Model
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Load dataset
data_path = 'https://raw.githubusercontent.com/Enqey/Diabetes_ml/main/diabetes.csv'
try:
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Display dataset information
st.subheader('Dataset Overview')
st.write(df.head())
st.write("Shape of Dataset:", df.shape)
st.write("Missing Values:", df.isnull().sum())

# Drop unnecessary column
if 'SkinThickness' in df.columns:
    df.drop('SkinThickness', axis=1, inplace=True)

# Split data into features and target
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Normalize features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Handle class imbalance
smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

# Train-Test Split with Stratification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)

# Function to get user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 29)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function (DPF)', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

    # Store user data in a DataFrame
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DPF,
        'Age': Age
    }

    # Transform into DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Function to log user data
def log_user_input(user_data, prediction):
    log_file = 'https://github.com/Enqey/Diabetes_ml/main/user_input_log.csv'
    user_data['Prediction'] = prediction  # Add prediction to the user data
    user_data['Timestamp'] = pd.Timestamp.now()  # Add timestamp for logging

    # Check if log file exists
    if os.path.exists(log_file):
        # Append to the existing file
        pd.DataFrame(user_data).to_csv(log_file, mode='a', index=False, header=False)
    else:
        # Create a new log file
        pd.DataFrame(user_data).to_csv(log_file, index=False)

# Get user input
user_input = get_user_input()

# Normalize user input using the same scaler
user_input_normalized = scaler.transform(user_input)

# Display user input
st.subheader('User Input:')
st.write(user_input)

# Random Forest Classifier with GridSearchCV
params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate model
y_pred = best_model.predict(x_test)
st.subheader('Model Performance:')
st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
st.write(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
st.write(f"AUC-ROC: {roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1]):.2f}")

# Display feature importance
st.subheader('Feature Importance:')
feature_importance = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': best_model.feature_importances_})
st.bar_chart(feature_importance.set_index('Feature'))

# Predict user input
prediction = best_model.predict(user_input_normalized)
result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# Log the user input
log_user_input(user_input, result)

# Display classification result
st.subheader('Classification:')
st.write(result)
