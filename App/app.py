import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import sklearn
from xgboost import XGBRegressor

st.header("Calorie burnt prediction")
col1, col2 = st.columns(2)
gender = option_menu("Gender", ["Male", "Female"], icons=["male", "female"], orientation="horizontal")

g = -1
if gender == "Male":
    g = 0
else:
    g = 1

with col2:
    Age = st.number_input("Age")
with col1:
    Height = st.number_input("Height")
with col2:
    Weight = st.number_input("Weight")
with col1:
    Duration = st.number_input("Duration")
with col2:
    Heart_Rate = st.number_input("Heart_Rate")
with col1:
    Body_temp = st.number_input("Body_temp")


def features():
    feature = pd.DataFrame({
        "Gender": g,
        "Age": Age,
        "Height": Height,
        "Heart_rate": Heart_Rate,
        "Body_temp": Body_temp,
        "Weight": Weight,
        "Duration": Duration

    }, index=[0])
    return feature


df = features()
model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(df)
if st.button("Predict"):
    st.header(prediction)
