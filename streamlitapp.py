import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model (replace with the actual path to your model file)
model = joblib.load("model.pkl")

# Define the feature names (same as used in training)
feature_names = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
    'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 
    'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18',
    'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24',
    'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30',
    'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36',
    'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42',
    'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48',
    'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54',
    'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60'
]

# Streamlit UI elements
st.title('Rock or Mine Prediction')

st.write(
    "This app predicts if the object is a Rock or a Mine based on feature inputs."
)

# Take input from the user
input_data = []
for feature in feature_names:
    value = st.number_input(f'Input value for {feature}', min_value=0.0, max_value=1.0, value=0.0, step=0.0001)
    input_data.append(value)

# Convert input data to numpy array
input_data_array = np.asarray(input_data).reshape(1, -1)

# Convert numpy array to pandas DataFrame (optional but useful for prediction clarity)
input_df = pd.DataFrame(input_data_array, columns=feature_names)

# When the user clicks the 'Predict' button
if st.button('Predict'):
    prediction = model.predict(input_df)  # Predict using the trained model
    if prediction[0] == 'R':
        st.write("Prediction: **Rock**")
    else:
        st.write("Prediction: **Mine**")
