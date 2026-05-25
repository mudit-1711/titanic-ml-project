import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

pclass = st.number_input("Passenger Class", 1, 3)
age = st.number_input("Age", 1, 100)
fare = st.number_input("Fare", 0)

if st.button("Predict"):

    data = np.array([[pclass, age, fare]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Survived")
    else:
        st.error("Not Survived")