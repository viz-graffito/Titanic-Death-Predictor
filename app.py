import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('models/standard_scaler.pkl', 'rb'))
encoder_ = load(open('models/encoder_.pkl', 'rb'))
svc_model = load(open('models/svc_model.pkl', 'rb'))

pclass = st.text_input("Pclass", placeholder="Enter Pclass")
gender = st.text_input("gender", placeholder="Enter Sex")
age = st.text_input("age", placeholder="Enter age")
sib = st.text_input("sib", placeholder="SibSp")
parch = st.text_input("parch", placeholder="Parch")
embarked = st.text_input("embarked", placeholder="Embarked")

btn_click = st.button("Predict")

if btn_click == True:
    if pclass and gender and age and sib and parch and embarked:
        age_r = np.array([float(age)]).reshape(-1, 1)
        age_transformed = scaler.transform(age_r)
        cat = [[int(pclass), str(gender), int(sib), int(parch), str(embarked)]]
        cat_tr = encoder_.transform(cat)
        cat_tr_r = np.array(cat_tr)
        query_point_final = np.concatenate((age_transformed, cat_tr_r), axis=1)
        pred = svc_model.predict(query_point_final)
        if pred == [1]:
            st.success('You are going to survive..')
        else:
            st.warning('You are going to die..')
    else:
        st.error("Enter the values properly.")