# Streamlit webapp
import keras
import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


model = keras.models.load_model("model.h5")

# Recreate encoders instead of loading corrupted pickles
label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(['Female', 'Male'])

one_hot_encoder_geography = OneHotEncoder()
one_hot_encoder_geography.fit([['France'], ['Spain'], ['Germany']])

with open("scaler.pkl", "rb") as file:
  scaler = pkl.load(file)
  

# Streamlit 
st.title("Customer Churn Prediction")


# Taking inputs from the user
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])

gender = st.selectbox("Gender", ['Female', 'Male'])

age = st.slider("Age", 18, 92)

balance = st.number_input("Balance")

credit_score = st.number_input("Credit Score")

estimated_salary = st.number_input("Estimated Salary")

tenure = st.slider("Tenure", 0, 10)

num_of_products = st.slider("Number of products", 1, 4)

has_cr_card = st.selectbox("Has Credit Card", [0, 1])

is_active_member = st.selectbox("Is Active Member", [0, 1])


# Make the individual inputs into a dataframe
input_data = pd.DataFrame({
  'CreditScore': [credit_score],
  "Gender": [label_encoder_gender.transform([gender])[0]],
  "Age": [age],
  "Tenure": [tenure],
  "Balance": [balance],
  "NumOfProducts": [num_of_products],
  "HasCrCard": [has_cr_card],
  "IsActiveMember": [is_active_member],
  "EstimatedSalary": [estimated_salary]
})

geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

# Combine the above with main df
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction
pred = model.predict(input_data_scaled)
pred_prob = pred[0][0]

st.write("Probability: ", pred_prob)

if pred_prob > 0.5:
  st.write("Customer is likely to churn")
else:
  st.write("Customer is unlikely to churn")