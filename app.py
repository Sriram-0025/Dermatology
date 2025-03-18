import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

st.title("Dermatology Class Prediction")

# Load the dataset for scaling
data_path = 'data_cleaned_dermatology.csv'
if not os.path.exists(data_path):
    st.error(f"Dataset file not found at {data_path}")
    st.stop()

# Load data
data = pd.read_csv(data_path)

# Load the model from Google Drive
model_path = 'my_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

loaded_model = load_model(model_path)

# Extract features for scaling
features = ['thinning of the suprapapillary epidermis', 'clubbing of the rete ridges',
            'spongiosis', 'fibrosis of the papillary dermis', 'koebner phenomenon',
            'elongation of the rete ridges', 'exocytosis', 'melanin incontinence',
            'pnl infiltrate', 'saw-tooth appearance of retes']

if not all(feature in data.columns for feature in features):
    st.error("Some required features are missing from the dataset.")
    st.stop()

X = data[features].values

# Initialize and fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Class labels mapping
class_labels = {
    1: 'psoriasis',
    2: 'seboreic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'chronic dermatitis',
    6: 'pityriasis rubra pilaris'
}

# Input form
with st.form(key='input_form'):
    inputs = {feature: st.number_input(feature, min_value=0, max_value=3, value=0) for feature in features}
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Convert input to numpy array
        new_input_data = np.array([list(inputs.values())])

        # Scale input
        new_input_data_scaled = scaler.transform(new_input_data)

        # Make prediction
        predictions = loaded_model.predict(new_input_data_scaled)
        predicted_class_index = np.argmax(predictions, axis=-1) + 1  # Adjust index to match class labels

        # Get predicted class label
        predicted_class_label = class_labels.get(predicted_class_index[0], "Unknown Class")

        # Display results
        st.write(f"Predicted class index: {predicted_class_index[0]}")
        st.write(f"Predicted class label: {predicted_class_label}")
