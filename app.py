import streamlit as st
import pandas as pd
import os

# Import Pandas Profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Pycaret for both classificaton and regression
from pycaret.classification import *
from pycaret.regression import *

with st.sidebar:
    st.image("1QOS8cNI-A61sKwLJ8Nf8Iw-e1532341850127.png")
    st.title("AutoML")
    choice=st.radio("Navigation",["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret")

# After saving the train_data.csv into data.csv 
if os.path.exists("data.csv"):
    df=pd.read_csv("data.csv",index_col=None)


if choice=="Upload":
    st.title("Upload Your Data for Modeling!")
    file=st.file_uploader("Upload a CSV file")
    if file:
        df=pd.read_csv(file, index_col=None)
        df.to_csv("data.csv",index=None)
        st.dataframe(df.head())
         

if choice=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report=df.profile_report()
    st_profile_report(profile_report)
    

if choice=="ML":
    st.title("Automated Machine Learning")
    target=st.selectbox("Select Target Column",df.columns)
    if st.button("Train Model"):
        setup(df,target=target)
        setup_df=pull()
        st.info("This is the ML experiment setup")
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")
       

if choice=="Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model",f,"trained_model.pkl")
