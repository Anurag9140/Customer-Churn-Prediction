import streamlit as st
import pandas as pd
import tensorflow
import pickle


model = tensorflow.keras.models.load_model("/Users/anuragverma/Desktop/NLP/DL/train_model.h5")

with open("/Users/anuragverma/Desktop/NLP/DL/ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("/Users/anuragverma/Desktop/NLP/DL/label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("/Users/anuragverma/Desktop/NLP/DL/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

    


st.set_page_config(page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered")

st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(45deg, #ff6f61, #6b5b95, #88b04b, #f7b7a3);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main {
        background-color:;
        padding: 2rem;
        border-radius: 15px;
    }
    .card {
        background-color:green;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .title {
        color:red;
        font-size: 35px;
        text-align: center;
        margin-bottom: 30px;
        
    }
    body{
        background-color:yellow;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title"><h1>Customer Churn Prediction</h1></div>', unsafe_allow_html=True)



st.markdown('<div class="card">', unsafe_allow_html=True)
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
age = st.slider('Age', 18, 92, 30)
credit_score = st.number_input('Credit Score', value=650)
balance = st.number_input('Balance', value=50000.0)
estimated_salary = st.number_input('Estimated Salary', value=70000.0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
st.markdown('</div>', unsafe_allow_html=True)


if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = ohe.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prob = prediction[0][0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(f'🔍 Churn Probability: **{prob*100:.2f}%**')

    if prob > 0.5:
        st.error("The customer is **likely to churn.")
    else:
        st.success(" The customer is **not likely to churn.**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
