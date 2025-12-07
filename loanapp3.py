
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Option A — Upload and load a local .pkl file in Google Colab (no Google Drive)


import pickle

# Preload model from fixed path
MODEL_PATH = "Logistic_Loans (4).pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# 2) Load the model from the uploaded bytes (no Drive involved)
import io
import pickle
import sklearn  # keep imported if the model uses scikit-learn classes

# Load directly from in-memory bytes
buffer = io.BytesIO(uploaded[filename])
model = pickle.load(buffer)

print("Model loaded:", type(model))

# 3) (Optional) Save a copy inside the Colab runtime (still not Google Drive)
# This lets you re-use the file in later cells via /content/Logistic_Loans.pkl
out_path = "/content/Logistic_Loans.pkl"
with open(out_path, "wb") as f:
    f.write(uploaded[filename])

print(f"Saved a runtime copy to: {out_path}")

# ---- Example: use the model (replace with your actual prediction code) ----
# import pandas as pd
# X_sample = pd.DataFrame({...})
# y_pred = model.predict(X_sample)


# Title for the app
# st.title("Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values
granted = st.slider("Granted Loan Amount (GRANTED)", min_value=1000, max_value=500000000, step=1000)
requested = st.slider("Requested Loan Amount (REQUESTED)", min_value=1000, max_value=50000000, step=1000)
fico = st.number_input("FICO Score (FICO)", min_value=0, max_value=900, step=1)
income = st.slider("Monthly Gross Income (INCOME)", min_value=0, max_value=1000000, step=1)
payment = st.slider("Monthly Housing Payment (PAYMENT)", min_value=0, max_value=50000000, step=1)
bounty = st.slider("Bounty (BOUNTY)", min_value=0, max_value=10000, step=50)


# Categorical inputs with options
reason = st.selectbox("Reason for Loan (REASON)", ["cover_an_unexpected_cost", "credit_card_refinancing", "home_improvement", "major_purchase", "other", "debt_conslidation"])
group = st.selectbox("FICO Score Group (GROUP)", ["fair", "poor", "good", "very_good", "Sales", "excellent"])
job = st.selectbox("Employment Status (JOB)", ["full_time", "part_time", "unemployed"])
sector = st.selectbox("Employement Sector (SECTOR)", ["consumer_discretionary", "information_technology", "energy", "consumer_staples", "communication_services", "materials", "utilities", "real_estate", "financials", "industrials"])
lender = st.selectbox("lender (LENDER)", ["A", "B", "C"])


# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "GRANTED": [granted],
    "REQUESTED": [requested],
    "FICO": [fico],
    "INCOME": [income],
    "PAYMENT": [payment],
    "BOUNTY": [bounty],
    "REASON": [reason],
    "GROUP": [group],
    "SECTOR": [sector],
    "LENDER": [lender],
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['REASON', 'JOB', 'GROUP', 'SECTOR', 'LENDER'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.write("The prediction is: **Bad Loan** 🚫")
    else:
        st.write("The prediction is: **Good Loan** 💲")



        """
What happens if the user enters a value not in the training data?

Example: User enters REASON = 'Vacation', but the model only knows 'DebtCon' and 'HomeImp'.

1. pd.get_dummies creates a new column: REASON_Vacation = 1.
2. The code then adds the *known* columns: REASON_DebtCon = 0 and REASON_HomeImp = 0.
3. The final filtering step *drops* the unknown REASON_Vacation column because it's not in the
   model's expected feature list.

Result: The model receives REASON_DebtCon = 0 and REASON_HomeImp = 0, which correctly
treats the unknown 'Vacation' input as "none of the known categories" (i.e., "Other").
"""
