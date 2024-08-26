
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title("Testing scikit-learn")

scaler = StandardScaler()
st.write("scikit-learn imported successfully!")

# Load the model from Google Drive
model_path = '/content/drive/MyDrive/DEEP LEARNING/saved_model/my_model.h5'
if os.path.exists(model_path):
    loaded_model = load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Load the dataset for scaling
data_path = '/content/drive/MyDrive/DEEP LEARNING/data_cleaned_dermatology.csv'
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    st.error(f"Dataset file not found at {data_path}")
    st.stop()

X = data[['thinning of the suprapapillary epidermis', 'clubbing of the rete ridges', 
          'spongiosis', 'fibrosis of the papillary dermis', 'koebner phenomenon', 
          'elongation of the rete ridges', 'exocytosis', 'melanin incontinence', 
          'pnl infiltrate', 'saw-tooth appearance of retes']].values

# Initialize the scaler
scaler = StandardScaler()
scaler.fit(X)

# Define your class labels mapping (1-based index)
class_labels = {
    1: 'psoriasis',
    2: 'seboreic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'chronic dermatitis',
    6: 'pityriasis rubra pilaris'
}

# Streamlit App
st.title("Dermatology Class Prediction")

# Input form
with st.form(key='input_form'):
    thinning = st.number_input('Thinning of the suprapapillary epidermis', min_value=0, max_value=3, value=0)
    clubbing = st.number_input('Clubbing of the rete ridges', min_value=0, max_value=3, value=0)
    spongiosis = st.number_input('Spongiosis', min_value=0, max_value=3, value=0)
    fibrosis = st.number_input('Fibrosis of the papillary dermis', min_value=0, max_value=3, value=0)
    koebner = st.number_input('Koebner phenomenon', min_value=0, max_value=3, value=0)
    elongation = st.number_input('Elongation of the rete ridges', min_value=0, max_value=3, value=0)
    exocytosis = st.number_input('Exocytosis', min_value=0, max_value=3, value=0)
    melanin = st.number_input('Melanin incontinence', min_value=0, max_value=3, value=0)
    pnl_infiltrate = st.number_input('Pnl infiltrate', min_value=0, max_value=3, value=0)
    saw_tooth = st.number_input('Saw-tooth appearance of retes', min_value=0, max_value=3, value=0)

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Create new input data
        new_input_data = np.array([[thinning, clubbing, spongiosis, fibrosis, koebner, elongation, exocytosis, melanin, pnl_infiltrate, saw_tooth]])

        # Scale the input data
        new_input_data_scaled = scaler.transform(new_input_data)

        # Make predictions
        predictions = loaded_model.predict(new_input_data_scaled)
        predicted_class_index = np.argmax(predictions, axis=-1) + 1  # Adjust index to match class starting from 1

        # Convert the predicted class index to a class label
        predicted_class_label = class_labels.get(predicted_class_index[0], "Unknown Class")

        # Display results
        st.write(f"Predicted class index: {predicted_class_index[0]}")
        st.write(f"Predicted class label: {predicted_class_label}")

