import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the model
model = tf.keras.models.load_model('model.h5')

## load encoder and scaler
with open('OHE_encoder_geo.pkl','rb') as file:
    label_geo_encoder=pickle.load(file)

with open('label_gender_encoder.pkl','rb') as file:
    label_gender_encoder=pickle.load(file)

with open('Scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', label_geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_gender_encoder.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

geo_encoded = label_geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_geo_encoder.get_feature_names_out(['Geography']))

input_data = pd.DataFrame(input_data)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_df_scaled = scaler.transform(input_data) 

## Prediction churn
prediction = model.predict(input_df_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")