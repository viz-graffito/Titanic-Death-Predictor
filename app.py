import streamlit as st

st.title('hello world')


btn_click = st.button('Click me')

if btn_click == True:
    st.write('you clicked me')
    st.balloons()


