# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:24:50 2021

@author: Enqey De-Ben Rockson
"""

import pandas as pd 
import streamlit as st 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

# Disclaimer section
st.write("""
         ***Disclaimer:***
         *Proof of Concept*

    This API is a test application and should not be used for self-diagnosis         
""")

# Load dataset
data_path = 'https://raw.githubusercontent.com/Enqey/Diabetes_ml/main/diabetes.csv'
try:
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Display dataset information
st.subheader('Data Information:')
st.dataframe(df)

st.write("""
 ***Statistical Indicators of Variables***

    Shows how each variable is significant in helping to predict your risk of diabetes         
""")

st.write(df.describe())

# Drop unnecessary column
df.drop('SkinThickness', axis=1, inplace=True)

# Split data into features and target
x = df.iloc[:, 0:6].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Function to get user input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function (DPF)', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

    # Store a dictionary into a variable 
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'BMI': BMI,
        'DPF': DPF,
        'Age': Age             
    }

    # Transform data into a DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store user input data into a variable 
user_input = get_user_input()

# Set subheader and display user's input 
st.subheader('User Input:')
st.write(user_input) 

# Create and train the model
RFC = RandomForestClassifier(random_state=42)
RFC.fit(x_train, y_train)

# Show model metrics 
st.subheader('Model Accuracy Score:')
st.write(f"{accuracy_score(y_test, RFC.predict(x_test)) * 100:.2f}%")

# Store model prediction in a variable 
pred = RFC.predict(user_input)

# Set a subheader and display classifier 
st.subheader('Classification:')
result = 'Diabetic' if pred[0] == 1 else 'Non-Diabetic'
st.write(result)
