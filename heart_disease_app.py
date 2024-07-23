import streamlit as st
import pandas as pd
import pickle
import os

def main():
    # Custom CSS for styling
    st.markdown(
        f"""
        <style>
        body {{
            background-color: black;
        }}
        .reportview-container {{
            background: black;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .sidebar .sidebar-content {{
            background: black;
            color: white;
        }}
        .stButton button {{
            background: linear-gradient(to right, #00f260, #0575e6);
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 20px;
        }}
        .stNumberInput, .stSelectbox, .stTextInput {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 10px;
        }}
        .stNumberInput > div > div > input, .stTextInput > div > div > input {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Heart Disease Prediction")

    # Input fields for each feature
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ['Male', 'Female'])
    chest_pain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
    rest_ecg = st.selectbox("Resting ECG", ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ['Upsloping', 'Flat', 'Downsloping'])

    # Encode categorical variables
    data = {
        'Age': age,
        'Sex': 1 if sex == 'Male' else 0,
        'ChestPainType': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}[chest_pain],
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == 'Yes' else 0,
        'RestingECG': {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[rest_ecg],
        'MaxHR': max_hr,
        'ExerciseAngina': 1 if exercise_angina == 'Yes' else 0,
        'Oldpeak': oldpeak,
        'ST_Slope': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}[st_slope]
    }
    
    input_data = pd.DataFrame([data])
    
    if st.button("Predict"):
        # Absolute paths to model and scaler files
        model_path = 'final_svm_model.sav'
        scaler_path = 'scaler.sav'
        
        try:
            loaded_model = pickle.load(open(model_path, 'rb'))
            loaded_scaler = pickle.load(open(scaler_path, 'rb'))
        except FileNotFoundError:
            st.error(f"Model or scaler files not found. Please make sure the files 'final_svm_model.sav' and 'scaler.sav' are present in the directory.")
            return
        
        # Standardize the features
        input_data_scaled = pd.DataFrame(loaded_scaler.transform(input_data), columns=input_data.columns)
        
        # Predict
        prediction = loaded_model.predict(input_data_scaled)

        # Display result
        if prediction[0] == 1:
            st.write("The model predicts that the person **has** heart disease.")
        else:
            st.write("The model predicts that the person **does not have** heart disease.")

if __name__ == "__main__":
    main()
