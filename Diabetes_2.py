# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:24:50 2021

@author: Enqey De-Ben Rockson
"""

import pandas as pd 
#import numpy as np
import streamlit as st 
from PIL import Image 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

       
#ic = st.file_uploader("‪‪D:\\pics\data-analytics - Copy.png", type = "png")

#if pic is not None:
    #convert the file to an opencv image.  
#    file_bytes = np.array(bytearray(pic.read()),dtype=np.uint8)
#    image = cv2.imdecode(file_bytes,1)
#    st.image(image,channels = "BGR", caption = 'Test your risk of diabetes', use_column_width = True)

#im = Image.open('D:\\pics\project concepts\diabetes.png')
st.image(im,use_column_width = True)

st.write("""
         ***Disclaimer: ***
         *Proof of Concept*
 
    This API is a test application and should not be used for self diagnosis         
         
         """)


df = pd.read_csv('D:\Docs\ACAD\DATA SCIENCE\Python\python - Data analysis\diabetes.csv')

st.subheader('Data information:')

st.dataframe(df)

st.write("""
 ***Statistical Indicators of variables***

    Shows how each variable is significant in helping to predict your risk of diabetes         
         
         """)


st.write(df.describe())

#chart = st.bar_chart(df['glucose'])



df.drop('SkinThickness', axis = 1, inplace = True)

x = df.iloc[:,0:6].values
y = df.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)



def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_thickness',0,122,72)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,81,29)
    
    #store a dictionary into a variable 
    user_data = {'pregnancies': pregnancies,
             'glucose': glucose ,
             'blood_pressure': blood_pressure,
             'BMI': BMI,
             'DPF': DPF,
             'Age': Age             
             }

    #transform data into database
    features = pd.DataFrame(user_data, index = [0])
    return features

#store user input data into a variable 
user_input = get_user_input()

#set subheader and display users input 
st.subheader ('user_input:')
st.write(user_input) 

#create model 
RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)

#show model metrics 
st.subheader ('Model Accuracy Score:')
st.write(str(accuracy_score(y_test, RFC.predict(x_test))*100) + '%')

#store model prediction in variable 
pred = RFC.predict(user_input)

#set a subheader and display classifer 
st.subheader('Classification:')
st.write(pred)
