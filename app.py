import streamlit as st
from src.prediction import InusrancePrediction

st.title("Inusrance Prediction")
st.write("Description about your proejct")

age = st.number_input("Enteger age: ")
annual_income_lpa = st.number_input("Enter annual income lpa")
policy_term_years = st.number_input("Enter policy term years")
sum_assured_lakhs = st.number_input("Enter sum assured lakhs")


if st.button('predict'):
    model = InusrancePrediction()
    result = model.prediction(age, annual_income_lpa, policy_term_years, sum_assured_lakhs)
    st.success(result)