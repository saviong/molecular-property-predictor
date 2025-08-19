# train.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

print("--- Script Started: Training Molecular Property Predictor ---")

# --- 1. Load and Prepare Data ---
print("Step 1: Loading and preparing data...")
try:
    # Adjust the column names based on the actual CSV file
    df = pd.read_csv('data/delaney_solubility.csv')
    df.rename(columns={'measured log(solubility:mol/L)': 'logS', 'smiles': 'SMILES'}, inplace=True, errors='ignore')
    # Use the correct column name for SMILES, it might be different in your file
    smiles_column_name = next((col for col in df.columns if 'smile' in col.lower()), None)
    if not smiles_column_name:
        raise ValueError("SMILES column not found in the dataset.")
    df.rename(columns={smiles_column_name: 'SMILES'}, inplace=True)
except FileNotFoundError:
    print("Error: delaney_solubility.csv not found in the 'data' directory.")
    exit()

print(f"Data loaded successfully with {len(df)} records.")

# --- 2. Featurization using RDKit ---
print("Step 2: Featurizing molecules using RDKit descriptors...")

def generate_descriptors(smiles_string):
    """
    Generate a dictionary of RDKit descriptors for a given SMILES string.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    # Calculate all available descriptors in RDKit
    descriptors = Descriptors.CalcMolDescriptors(mol)
    return descriptors

# Apply the function to the SMILES column
descriptor_data = df['SMILES'].apply(generate_descriptors)
descriptor_df = pd.DataFrame(list(descriptor_data))

# Handle potential invalid SMILES that returned None
initial_count = len(df)
valid_indices = descriptor_df.notna().all(axis=1)
descriptor_df = descriptor_df[valid_indices].reset_index(drop=True)
df_clean = df[valid_indices].reset_index(drop=True)
print(f"Removed {initial_count - len(df_clean)} invalid SMILES strings.")

# Combine original data with new features
df_final = pd.concat([df_clean[['SMILES', 'logS']], descriptor_df], axis=1)

# Prepare data for model training
X = df_final.drop(['SMILES', 'logS'], axis=1)
y = df_final['logS']

# Save the list of feature names (this is crucial for the app)
features_list = X.columns.tolist()
with open('features.json', 'w') as f:
    json.dump(features_list, f)
print("Feature list saved to features.json")


# --- 3. Train the Model ---
print("Step 3: Training the RandomForestRegressor model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training completed.")

# --- 4. Evaluate the Model ---
print("Step 4: Evaluating the model performance...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  R-squared (R2): {r2:.4f}")

# --- 5. Save the Trained Model ---
print("Step 5: Saving the trained model to model.pkl...")
joblib.dump(model, 'model.pkl')
print("Model saved successfully.")

print("--- Script Finished ---")