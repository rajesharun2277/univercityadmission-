import streamlit as st
import pandas as pd
import numpy as np
import pickle

clf = pickle.load(open("case_study_university.pkl.pkl","rb"))

def predict(data):
    clf = pickle.load(open("case_study_university.pkl.pkl","rb"))
    return clf.predict(data)

st.title("Advertising Spends Prediction Using Machine Learning")
st.markdown("This Model Identify Total Spends On Advertising")

st.header("Advertising Spend On Various Media")
col1,col2 = st.columns(2)

with col1:
    st.text("TV")
    TV = st.slider("Advertising Spend on TV", 1.0, 10000.0, 0.5)
    st.text("Radio")
    Radio = st.slider("Advertising Spend on Radio", 1.0, 10000.0, 0.5)
    st.text("NewsPaper")
    NewsPaper = st.slider("Advertising Spend on NewsPaper", 1.0, 10000.0, 0.5)
                          
st.text('')
if st.button("Sales Prediction"):
    result= clf.predict(np.array([[TV,Radio,NewsPaper]]))
    st.text(result[0])
    
st.markdown("Developed  at NIELIT Daman")
                  
