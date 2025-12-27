import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("wine_model.pkl", "rb"))

st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="centered")
st.title("🍷 Wine Quality Predictor")

# -------- INPUTS (ALL FLOATS) --------
alcohol = st.number_input("Alcohol",
                          min_value=0.0, max_value=20.0, value=7.9, step=0.1)

malic_acid = st.number_input("Malic Acid",
                             min_value=0.0, max_value=10.0, value=0.35, step=0.01)

ash = st.number_input("Ash",
                      min_value=0.0, max_value=10.0, value=0.46, step=0.01)

alcalinity = st.number_input("Alcalinity of Ash",
                             min_value=0.0, max_value=50.0, value=3.6, step=0.1)

magnesium = st.number_input("Magnesium",
                             min_value=0.0, max_value=200.0, value=0.078, step=0.001)

phenols = st.number_input("Total Phenols",
                           min_value=0.0, max_value=50.0, value=15.0, step=0.1)

flavanoids = st.number_input("Flavanoids",
                              min_value=0.0, max_value=50.0, value=37.0, step=0.1)

nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols",
                                       min_value=0.0, max_value=2.0, value=0.9973, step=0.0001)

proanthocyanins = st.number_input("Proanthocyanins",
                                   min_value=0.0, max_value=10.0, value=3.35, step=0.01)

color_intensity = st.number_input("Color Intensity",
                                   min_value=0.0, max_value=20.0, value=0.86, step=0.01)

hue = st.number_input("Hue",
                       min_value=0.0, max_value=15.0, value=12.8, step=0.1)

# -------- PREDICTION --------
if st.button("Predict Wine Quality"):
    sample = np.array([[alcohol, malic_acid, ash, alcalinity, magnesium,
                        phenols, flavanoids, nonflavanoid_phenols,
                        proanthocyanins, color_intensity, hue]])

    prediction = model.predict(sample)[0]
    st.success(f"🍷 Predicted Wine Quality: {prediction}")
