# **1. Imports**
# Import necessary libraries for data handling, preprocessing, model training, evaluation, and saving the model.
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Optional: Tools for hyperparameter tuning and visualization
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shap


# **2. Set Up Logging**
# Configure logging to debug the mapping issues
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# **3. Load and Explore Data**
# Load the dataset and perform initial exploration to understand its structure and quality.
data = pd.read_csv('Data/mds_ed.csv', low_memory=False)

# Basic data exploration
print(data.head())  # Check the first few rows
print(data.info())  # Check data types and missing values

# Fill missing values for numeric columns with the mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# **4. Feature and Target Selection**
# Define the features (`X`) and target variable (`y`) based on the dataset.
# Select features   
X = data[[
    'demographics_gender', 'demographics_age', 'demographics_ethnicity_asian',
    'demographics_ethnicity_black/african', 'demographics_ethnicity_hispanic/latino',
    'demographics_ethnicity_other', 'demographics_ethnicity_white', 
    'vitals_temperature_mean', 'vitals_heartrate_mean', 'vitals_resprate_mean',
    'vitals_resprate_median', 'vitals_o2sat_mean', 'vitals_sbp_mean',
    'vitals_dbp_mean', 'labvalues_alanine_aminotransferase_(alt)_mean',
    'labvalues_asparate_aminotransferase_(ast)_mean', 'labvalues_sodium_mean',
    'labvalues_potassium_mean', 'labvalues_c-reactive_protein_mean'
]]

# Set target variable as diagnoses_a41 and diagnoses_a419 columns combined
y = data[['diagnoses_a41', 'diagnoses_a419']].max(axis=1)  # Use max to indicate presence of sepsis (1 if either column is 1)

# **5. Handle Class Imbalance with SMOTE**
# Apply SMOTE to handle class imbalance if more than one class is present
if len(np.unique(y)) > 1:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    X_resampled, y_resampled = X, y

# **6. Train-Test Split**
# Split the data into training and test sets using stratified sampling to preserve class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# **7. Model Training with Cross-Validation**
# Train the XGBoost classifier with basic or tuned hyperparameters.
# Initialize the model
model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    eval_metric='mlogloss',
    random_state=42,
    tree_method= 'hist',  # Use GPU for training
    device= 'cpu'
)

# Perform cross-validation to check model performance
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Train the model on the full training set
model.fit(X_train, y_train)

# **8. Model Evaluation**
# Evaluate the model using accuracy, classification report, and confusion matrix.
# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Detailed evaluation
print(classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis']))

# **9. Hyperparameter Tuning (Optional)**
# Improve the model’s performance by fine-tuning its hyperparameters.
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, tree_method='hist', device='cpu'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# **10. Save and Deploy the Model**
# Save the trained model and create functions for user input and prediction.
joblib.dump(model, 'xgboost_model.pkl')

# Define preprocessing function
def preprocess_input(input_data):
    # Example preprocessing steps (should match the training preprocessing)
    # Ensure the input data is a DataFrame with the same features as training
    input_df = pd.DataFrame([input_data], columns=X.columns)
    # Additional preprocessing steps like scaling or encoding if needed
    input_df[numeric_cols] = input_df[numeric_cols].fillna(input_df[numeric_cols].mean())  # Fill any missing values for numeric columns
    return input_df

# Load and predict
def predict_diagnosis(input_data):
    processed_data = preprocess_input(input_data)  # Define preprocessing steps
    loaded_model = joblib.load('xgboost_model.pk2')
    predictions = loaded_model.predict(processed_data)
    decoded_predictions = ['No Sepsis' if pred == 0 else 'Sepsis' for pred in predictions]
    return decoded_predictions

# **11. Interpretability (Optional)**
# Use SHAP to interpret the model’s predictions.
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# **12. Final Notes**
# - Ensure robust preprocessing for missing data, scaling, or one-hot encoding if required.
# - Validate the pipeline on a separate validation set if your dataset is large.
# - Deploy the model with Flask, FastAPI, or Streamlit for user interaction.
