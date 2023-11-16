import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('./models/standard_scaler_2.pkl', 'rb'))
model = load(open('./models/classifier_2.pkl', 'rb'))

PCLASS = st.selectbox("Passenger Class",['1st','2nd','3rd'], placeholder="Enter your Ticket Class")
GENDER = st.selectbox("Gender",['Male','Female'], placeholder="Enter Your Gender")
AGE = st.number_input("Age", placeholder="Enter Your Age", step =1)
ALONE = st.radio("Are you alone?",['Yes','No'])

if ALONE == 'Yes':
    FAMILY = st.slider('What is your family size', 0, 11, 0, disabled=True)
else:
    FAMILY = st.slider('What is your family size', 0, 11, 0, disabled=False)

EMBARKED = st.selectbox("Embarked From",['Cherbourg','Queenstown','Southampton'], placeholder="From where do you embarked in Englad")

btn_click = st.button("Predict")

if btn_click == True:
    if PCLASS and GENDER and AGE and ALONE or FAMILY and EMBARKED:
        
        #setting up age
        AGE_F = np.array([float(AGE)]).reshape(1, -1)
        AGE_Var = scaler.transform(AGE_F)
        
        #setting up gender
        GEN_Var = 1 if GENDER == 'Male' else 0
        
        #setting up Pclass
        def pCls(class_1):
            emb = {'1st': 1, '2nd': 2, '3rd':3}
            if class_1 == '1st':
                return emb['1st']
            elif class_1 == '2nd':
                return emb['2nd']
            else:
                return emb['3rd'] 
            
        PCLASS_Var = pCls(PCLASS)

        #setting up Embarked
        def EMB(embarked):
            emb = {'C': 0, 'Q': 1, 'S':2}
            if embarked == 'Q':
                return emb['Q']
            elif embarked == 'S':
                return emb['S']
            else:
                return emb['C']

        EMBARKED_VAR = EMB(EMBARKED)

        # setting up alone
        ALONE_VAR = 1 if ALONE == 'Yes' else 0

        # setting up family
        FAMILY_VAR = int(FAMILY)

        INP_1 = np.array([GEN_Var,PCLASS_Var,EMBARKED_VAR,ALONE_VAR,FAMILY_VAR]).reshape(1,-1)
        
        INPUT = np.concatenate((AGE_Var,INP_1), axis=1 ) 
        PREDICTION = model.predict(INPUT)
        if PREDICTION == [1]:
            st.success('You are going to survive..')
        else:
            st.warning('You are going to die..')
    else:
        st.error("Enter the values properly.")
