import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('wine_quality_model.pkl')

# Streamlit UI
st.title("Wine Quality Prediction")

# Dictionary to hold feature descriptions
feature_descriptions = {
    'fixed acidity': 'Acidity that remains after the wine is fermented. Example: Higher acidity can make the wine taste more tart.',
    'volatile acidity': 'A measure of the amount of acetic acid in the wine. Example: Too much volatile acidity can lead to a vinegar taste.',
    'citric acid': 'A natural preservative found in citrus fruits. Example: Adds freshness and flavor to the wine.',
    'residual sugar': 'The amount of sugar remaining after fermentation. Example: Higher residual sugar can make the wine taste sweeter.',
    'chlorides': 'A measure of salt in the wine. Example: High levels can affect the taste and quality.',
    'free sulfur dioxide': 'A measure of the amount of free SO2 in the wine. Example: Helps prevent oxidation and spoilage.',
    'total sulfur dioxide': 'The total amount of SO2 in the wine. Example: High levels can lead to off-flavors.',
    'density': 'The density of the wine, which can indicate sugar and alcohol content. Example: Higher density can indicate higher sugar levels.',
    'pH': 'A measure of acidity or alkalinity. Example: A lower pH means higher acidity.',
    'sulphates': 'A measure of the amount of sulfates in the wine. Example: Can enhance the flavor and stability of the wine.',
    'alcohol': 'The percentage of alcohol in the wine. Example: Higher alcohol content can lead to a stronger flavor.',
}

features = []
for feature in feature_descriptions.keys():
    value = st.number_input(feature, help=feature_descriptions[feature])
    features.append(value)

if st.button("Predict"):
    prediction = model.predict([features])
    st.write(f"Predicted Quality: {prediction[0]}")