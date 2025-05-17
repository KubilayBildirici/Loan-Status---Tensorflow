import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

# Modeli yükleme, cache_resource ile önbellekle
@st.cache_resource
def load_model_1_resource(path='credit_score_model/credit_score_model.h5'):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_model_2_resource(path='loan_status_model/loan_status_model.h5'):
    return tf.keras.models.load_model(path)

# Manuel transformer oluşturma, cache_data ile önbellekle
@st.cache_data
def load_transformer(data_path='loan_data.csv'):
    df = pd.read_csv(data_path)
    df.drop('loan_status', axis=1, inplace=True)
    X = df.drop('credit_score', axis=1)
    ct = make_column_transformer(
        (MinMaxScaler(), [
            'person_age', 'person_income', 'person_emp_exp',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length'
        ]),
        (OneHotEncoder(handle_unknown='ignore'), [
            'person_gender', 'person_education',
            'person_home_ownership', 'loan_intent',
            'previous_loan_defaults_on_file'
        ])
    )
    ct.fit(X)
    return ct

ct = load_transformer()

page_bg_img = '''
<style>
.stApp {
  background-image: url("https://wallpaperaccess.com/full/1567674.jpg");
  background-size: cover;
  background-position: top left;
  background-repeat: no-repeat;
  background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app
st.title("Credit Score Predictor")
st.write("Enter the applicant's details in the sidebar and click to predict.")

# About section
st.sidebar.header("About")
st.sidebar.write(
    "Project By Kubilay Bildirici"
)

# Sidebar inputs
age = st.sidebar.number_input("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["female", "male"])
education = st.sidebar.selectbox("Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
income = st.sidebar.number_input("Annual Income", 0.0, 1e6, 50000.0)
emp_exp = st.sidebar.number_input("Years Employed", 0, 50, 5)
home = st.sidebar.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
loan_amnt = st.sidebar.number_input("Loan Amount", 0.0, 1e6, 10000.0)
intent = st.sidebar.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
rate = st.sidebar.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
hist_len = st.sidebar.number_input("Credit History Length (years)", 0.0, 50.0, 5.0)
prev_default = st.sidebar.selectbox("Previous Defaults", ["No", "Yes"])

# Derived feature
loan_percent_income = loan_amnt / income

# Prepare input DataFrame for credit score model
input_df1 = pd.DataFrame({
    'person_age': [age],
    'person_gender': [gender],
    'person_education': [education],
    'person_income': [income],
    'person_emp_exp': [emp_exp],
    'person_home_ownership': [home],
    'loan_amnt': [loan_amnt],
    'loan_intent': [intent],
    'loan_int_rate': [rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [hist_len],
    'previous_loan_defaults_on_file': [prev_default]
})

st.subheader("Input Data")
st.write(input_df1)

model_1 = load_model_1_resource()  # credit score model

# Preprocess and predict
tab = ct.transform(input_df1)
y_pred = model_1.predict(tab).squeeze()

st.subheader("Predicted Credit Score")
st.write(f"{y_pred:.2f}")

if(y_pred >= 300 and y_pred <= 579):
    st.write("Very Poor Credit Score")
if(y_pred >= 580 and y_pred <= 669):
    st.write("Fair Credit Score")
if(y_pred >= 670 and y_pred <= 739):
    st.write("Good Credit Score")
if(y_pred >= 740 and y_pred <= 799):
    st.write("Very Good Credit Score")
if(y_pred >= 800 and y_pred <= 850):
    st.write("Exceptional Credit Score")



# Manuel transformer oluşturma, cache_data ile önbellekle
@st.cache_data
def load_transformer_2(data_path='loan_data.csv'):
    df = pd.read_csv(data_path)
    X = df.drop("loan_status", axis=1)
    ct2 = make_column_transformer(
        # Sayısal özellikler
        (MinMaxScaler(), [
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score"
        ]),
        # Kategorik özellikler
        (OneHotEncoder(handle_unknown="ignore"), [
            "person_gender",
            "person_education",
            "person_home_ownership",
            "loan_intent",
            "previous_loan_defaults_on_file"
        ])
    )

    ct2.fit(X)
    return ct2

ct2 = load_transformer_2()

# — 2) İkinci model: Loan status —
input_df2 = input_df1.copy()
input_df2['credit_score'] = y_pred


model_2 = load_model_2_resource() #loan status model

#prepocess and predict
tab_2 = ct2.transform(input_df2)


y_pred_2 = model_2.predict(tab_2).squeeze()


status = "Approved" if y_pred_2 >= 0.5 else "rejected"
st.metric("📊 Predicted Loan Status", status)