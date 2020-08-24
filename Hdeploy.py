import streamlit as st
import os
import pandas as pd
import numpy as np
import dill
from sklearn.base import BaseEstimator, TransformerMixin
from feat_engin import split_data,create_new_features
from PIL import Image


## Title
st.title("MLOps House Price Prediction")
st.sidebar.title("Interactive House Price Predictor")
img=Image.open("homevalue.jpg")
st.sidebar.image(img, width=200)

st.header("Enter Input Values")
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Median age of a house within a block',  value=0, format= "%d",min_value=0, max_value=1000)
total_rooms = st.number_input('Total number of rooms', value=0, format= "%d",min_value=0, max_value=100000)
total_bedrooms = st.number_input('Total number of bed rooms', value=0, format= "%d",min_value=0, max_value=100000)
population = st.number_input('Total number of people residing within a block', value=0, format= "%d",min_value=0, max_value=100000000)
households = st.number_input('Total number of households',value=0, format= "%d",min_value=0, max_value=1000000000)
median_income = st.number_input('Median income for households')
ocean_proximity = st.selectbox("Select your occupaion",["<1H OCEAN","INLAND","ISLAND","NEAR BAY","NEAR OCEAN"])

if st.sidebar.checkbox("Click here to get help for filling data"):
    st.sidebar.json({'longitude': -121.89,
                            'latitude': 37.29,
                            'housing_median_age': 38.0,
                            'total_rooms': 1568.0,
                            'total_bedrooms': 351.0,
                            'population': 710.0,
                            'households': 339.0,
                            'median_income': 2.7042,
                            'ocean_proximity': '<1H OCEAN'})

input_sample = pd.DataFrame({'longitude': longitude,
                            'latitude': latitude,
                            'housing_median_age': housing_median_age,
                            'total_rooms': total_rooms,
                            'total_bedrooms': total_bedrooms,
                            'population': population,
                            'households': households,
                            'median_income': median_income,
                            'ocean_proximity': ocean_proximity}, index=[0])

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

with open("E:/Pactera Edge/streamlit-demo-master/house_price_model.pkl", 'rb') as f:
    model = dill.load(f)

if st.button("Predict"):
    st.write('Predicted House Value')
    st.success(model.predict(input_sample)[0])
