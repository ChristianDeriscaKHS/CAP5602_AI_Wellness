# Imports
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
from sklearn.utils import resample

#2. Set Up Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#3. Load and Explore Data
data = pd.read_csv('Data/mds_ed.csv', low_memory=False)

# Basic data exploration
print(data.head())
print(data.info())

# Fill missing values for numeric columns with the mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

#Feature and Target Selection
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
y = data[['diagnoses_a41', 'diagnoses_a419']].max(axis=1)

#Stratified Oversampling by Ethnicity
ethnicity_columns = [
    'demographics_ethnicity_asian', 'demographics_ethnicity_black/african',
    'demographics_ethnicity_hispanic/latino', 'demographics_ethnicity_other',
    'demographics_ethnicity_white'
]

# List to store resampled data
X_resampled_list = []
y_resampled_list = []

# Loop through each ethnicity column and apply SMOTE or resampling within each group
for ethnicity_col in ethnicity_columns:
    # Filter the data for the current ethnicity group
    ethnicity_group = X[X[ethnicity_col] == 1]
    y_group = y[ethnicity_group.index]

    # Resample to ensure similar class distribution (Sepsis vs No Sepsis) within each group
    if len(np.unique(y_group)) > 1:
        sepsis_group = ethnicity_group[y_group == 1]
        non_sepsis_group = ethnicity_group[y_group == 0]

        if len(sepsis_group) < len(non_sepsis_group):
            sepsis_group_resampled = resample(sepsis_group,
                                              replace=True,
                                              n_samples=len(non_sepsis_group),
                                              random_state=42)
            X_group_resampled = pd.concat([sepsis_group_resampled, non_sepsis_group])
            y_group_resampled = pd.concat([y.loc[sepsis_group_resampled.index], y.loc[non_sepsis_group.index]])
        else:
            non_sepsis_group_resampled = resample(non_sepsis_group,
                                                  replace=True,
                                                  n_samples=len(sepsis_group),
                                                  random_state=42)
            X_group_resampled = pd.concat([sepsis_group, non_sepsis_group_resampled])
            y_group_resampled = pd.concat([y.loc[sepsis_group.index], y.loc[non_sepsis_group_resampled.index]])
    else:
        X_group_resampled, y_group_resampled = ethnicity_group, y_group

    # Append resampled data to the list
    X_resampled_list.append(X_group_resampled)
    y_resampled_list.append(y_group_resampled)

# Combine all resampled data
X_resampled = pd.concat(X_resampled_list, axis=0)
y_resampled = pd.concat(y_resampled_list, axis=0)

#Visualize Distribution After Stratified Oversampling
plt.figure(figsize=(14, 6))

#Original Ethnicity Distribution
plt.subplot(1, 2, 1)
original_counts = X[ethnicity_columns].sum()
original_counts.plot(kind='bar', color='skyblue')
plt.title('Original Ethnicity Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)

#Resampled Ethnicity Distribution
plt.subplot(1, 2, 2)
resampled_counts = X_resampled[ethnicity_columns].sum()
resampled_counts.plot(kind='bar', color='salmon')
plt.title('Resampled Ethnicity Distribution After Stratified Oversampling')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

#Model Training with Cross-Validation
model = xgb.XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    device='cpu'
)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy scores: {cv_scores}')
print(f'Mean cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Train the model on the full training set
model.fit(X_train, y_train)

#Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=['No Sepsis', 'Sepsis']))

#Hyperparameter Tuning
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

#Save and Deploy the Model
joblib.dump(model, 'xgboost_model.pkl3')

def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_df[numeric_cols] = input_df[numeric_cols].fillna(input_df[numeric_cols].mean())
    return input_df

def predict_diagnosis(input_data):
    processed_data = preprocess_input(input_data)
    loaded_model = joblib.load('xgboost_model.pkl3')
    predictions = loaded_model.predict(processed_data)
    decoded_predictions = ['No Sepsis' if pred == 0 else 'Sepsis' for pred in predictions]
    return decoded_predictions

#Interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

#Final Notes
# - Ensure robust preprocessing for missing data, scaling, or one-hot encoding if required.
# - Validate the pipeline on a separate validation set if your dataset is large.
# - Deploy the model with Flask, FastAPI, or Streamlit for user interaction.
