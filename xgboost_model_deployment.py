# Import necessary libraries for deployment
import joblib
import pandas as pd
import streamlit as st

# **1. Load the Model**
# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')

# **2. Define Preprocessing Function**
# Define a function to preprocess input data
def preprocess_input(input_data):
    # Ensure the input data is a DataFrame with the same features as training
    columns = [
    'demographics_gender', 'demographics_age', 'demographics_ethnicity_asian',
    'demographics_ethnicity_black/african', 'demographics_ethnicity_hispanic/latino',
    'demographics_ethnicity_other', 'demographics_ethnicity_white', 
    'vitals_temperature_mean', 'vitals_heartrate_mean', 'vitals_resprate_mean',
    'vitals_resprate_median', 'vitals_o2sat_mean', 'vitals_sbp_mean',
    'vitals_dbp_mean', 'labvalues_alanine_aminotransferase_(alt)_mean',
    'labvalues_asparate_aminotransferase_(ast)_mean', 'labvalues_sodium_mean',
    'labvalues_potassium_mean', 'labvalues_c-reactive_protein_mean'
    ]
    
    input_df = pd.DataFrame([input_data], columns=columns)
    
    # Convert gender to 0 or 1 based on input
    input_df['demographics_gender'] = input_df['demographics_gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Handle ethnicity conversion
    ethnicity = input_data['demographics_ethnicity']
    ethnicity_mapping = {
        'Asian': ['demographics_ethnicity_asian'],
        'Black/African American': ['demographics_ethnicity_black/african'],
        'Hispanic/Latino': ['demographics_ethnicity_hispanic/latino'],
        'White': ['demographics_ethnicity_white'],
        'Other': ['demographics_ethnicity_other']
    }
    
    # Set all ethnicity columns to 0
    for col in ethnicity_mapping.values():
        input_df[col[0]] = 0
    
    # Set the selected ethnicity column to 1
    if ethnicity in ethnicity_mapping:
        input_df[ethnicity_mapping[ethnicity][0]] = 1

    return input_df

# **3. Define Treatment Recommendation Function**
# Provide treatment recommendations if sepsis is suspected
def sepsis_treatment_recommendations():
    recommendations = """
    ### Recommended Treatment for Suspected Sepsis:
    
    **1. Administer Broad-Spectrum Antibiotics:**
       - Start immediately after blood cultures are drawn to cover a wide range of pathogens.

    **2. Fluid Resuscitation:**
       - Give an initial bolus of 30 mL/kg of crystalloids to support blood pressure and organ perfusion.

    **3. Blood Cultures and Laboratory Tests:**
       - Obtain blood cultures before administering antibiotics to identify causative organisms.
       - Measure serum lactate levels to assess tissue perfusion.

    **4. Monitor Vital Signs and Organ Function:**
       - Continuously monitor blood pressure, urine output, and mental status to assess patient response.

    **5. Vasopressors:**
       - If hypotension persists after fluid resuscitation, start vasopressors (e.g., norepinephrine) to maintain MAP ≥ 65 mmHg.

    **6. Contact Critical Care Team:**
       - Involve specialists early for further intervention and management.

    **Note:** Consult the hospital's sepsis management protocol for specific guidelines.
    """
    return recommendations

# **4. Streamlit Interface**
# Set up Streamlit app for user interaction
st.title("AI Wellness - Sepsis Prediction")

# User inputs for each feature
input_data = {
    'demographics_gender': st.selectbox('Gender', ['Male', 'Female']),
    'demographics_age': st.slider('Age', 0, 100, 25),
    'demographics_ethnicity': st.selectbox('Ethnicity', ['Asian', 'Black/African', 'Hispanic/Latino', 'Other', 'White']),
    'vitals_temperature_mean': st.number_input('Temperature Mean', value=38.0),
    'vitals_heartrate_mean': st.number_input('Heart Rate Mean', value=75.0),
    'vitals_resprate_mean': st.number_input('Respiratory Rate Mean', value=18.0),
    'vitals_resprate_median': st.number_input('Respiratory Rate Median', value=18.0),
    'vitals_o2sat_mean': st.number_input('O2 Saturation Mean', value=98.0),
    'vitals_sbp_mean': st.number_input('Systolic BP Mean', value=120.0),
    'vitals_dbp_mean': st.number_input('Diastolic BP Mean', value=80.0),
    'labvalues_alanine_aminotransferase_(alt)_mean': st.number_input('ALT Mean', value=30.0),
    'labvalues_asparate_aminotransferase_(ast)_mean': st.number_input('AST Mean', value=30.0),
    'labvalues_sodium_mean': st.number_input('Sodium Mean', value=140.0),
    'labvalues_potassium_mean': st.number_input('Potassium Mean', value=4.0),
    'labvalues_c-reactive_protein_mean': st.number_input('C-Reactive Protein Mean', value=5.0),
}

# **5. Make Prediction**
# Button to make prediction
if st.button('Predict Sepsis'):
    input_df = preprocess_input(input_data)
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write('Prediction: Sepsis suspected')
        st.markdown(sepsis_treatment_recommendations())  # Display treatment recommendations
    else:
        st.write('Prediction: No Sepsis Suspected')

# **6. Deployment Notes**
# - Streamlit provides an easy way to create a GUI for the model.
# - To run the Streamlit app, use the command: `streamlit run xgboost_model_deployment.py`.
# - Consider using Docker to containerize the Streamlit app and deploy it on cloud platforms.
