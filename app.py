# app.py

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import joblib
import json
from PIL import Image
import io

# --- Configuration and Artifact Loading ---
st.set_page_config(page_title="Molecular Property Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    """Load the model and feature list."""
    model = joblib.load('model.pkl')
    with open('features.json', 'r') as f:
        feature_list = json.load(f)
    return model, feature_list

# --- UPDATED DATA LOADING ---
@st.cache_data
def load_searchable_data():
    """
    Load the full Delaney dataset and prepare it for searching.
    We will use the 'Compound ID' as the name.
    """
    df = pd.read_csv('data/delaney_solubility.csv')
    
    # CORRECTED PART: Use 'SMILES' (uppercase) from the CSV, 
    # and also rename it to 'smiles' (lowercase) for consistency in the app.
    df_search = df[['Compound ID', 'SMILES']].copy()
    df_search.rename(columns={'Compound ID': 'name', 'SMILES': 'smiles'}, inplace=True)
    
    return df_search

# Load all necessary files
model, feature_list = load_artifacts()
# Call the new data loading function
searchable_molecules_df = load_searchable_data()


# --- Featurization and Display Functions ---
def generate_descriptors(smiles_string):
    """Generate RDKit descriptors for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    descriptors = Descriptors.CalcMolDescriptors(mol)
    features_df = pd.DataFrame(0.0, index=[0], columns=feature_list)
    
    for desc_name, desc_value in descriptors.items():
        if desc_name in features_df.columns:
            features_df.loc[0, desc_name] = desc_value
            
    return features_df

def display_molecule(smiles_string):
    """Generates and displays an image of the molecule."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None

# --- Main App UI ---
st.title("🧪 Molecular Property Predictor")
st.write(
    "Predict the aqueous solubility (LogS) of a chemical by providing its SMILES string."
    " You can use the helper on the left to find a molecule from the training dataset or enter any valid SMILES string directly on the right."
)
st.markdown("---")

# --- UI Layout and Logic ---
col1, col2 = st.columns([1, 2])

# --- Column 1: Molecule Search Helper ---
with col1:
    st.header("Molecule Helper")
    
    # Text input for searching by Compound ID
    search_term = st.text_input("Search molecule by Compound ID:", "")
    
    # Filter the DataFrame based on the search term
    if search_term:
        filtered_df = searchable_molecules_df[searchable_molecules_df['name'].str.contains(search_term, case=False)]
    else:
        filtered_df = searchable_molecules_df
        
    molecule_names = ["Select a molecule..."] + sorted(filtered_df['name'].tolist())
    
    selected_name = st.selectbox(
        "Or select from the list (over 1100 available):",
        options=molecule_names
    )
    
    if selected_name != "Select a molecule...":
        selected_smiles = filtered_df[filtered_df['name'] == selected_name]['smiles'].iloc[0]
        st.session_state.smiles_input = selected_smiles
    

# --- Column 2: Prediction Area ---
with col2:
    st.header("Enter SMILES for Prediction")
    
    smiles_input = st.text_input(
        "SMILES String:",
        key="smiles_input",
        placeholder="Enter SMILES string here, e.g., CCO"
    )

    if st.button("Predict Solubility", type="primary"):
        if smiles_input:
            with st.spinner("Calculating..."):
                try:
                    features_df = generate_descriptors(smiles_input)
                    if features_df is not None:
                        prediction = model.predict(features_df)
                        predicted_logs = prediction[0]

                        st.metric(
                            label="Predicted Solubility (LogS)",
                            value=f"{predicted_logs:.4f}",
                            help="LogS is the logarithm of the molar solubility (mol/L). Higher is more soluble."
                        )
                        
                        st.subheader("Molecule Structure")
                        img = display_molecule(smiles_input)
                        if img:
                            st.image(img, caption=f"Structure for {smiles_input}")
                        else:
                            st.warning("Could not generate image for the provided SMILES.")
                    else:
                        st.error("Invalid SMILES string. Please check the input.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a SMILES string to get a prediction.")