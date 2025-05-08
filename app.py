import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open ('geo_encoder.pkl', 'rb') as file:
    geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



#streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_mumber = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_mumber],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded 'Geography' with the rest of the input
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


#Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict chrun
predict = model.predict(input_data_scaled)
prediction_proba = predict[0][0]

# Display the prediction
st.write('Prediction Probability:', prediction_proba)

if prediction_proba > 0.5:
    st.write('The coustomer is likely to chrun.')
else:
    st.write('The coustomer is not likely to churn.')