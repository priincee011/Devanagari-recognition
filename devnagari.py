import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import io
from tensorflow import keras
import pickle

st.title("Devanagari Character Recognition")

num_or_char = st.radio("Choose an option", ("Number", "Character"))

if num_or_char == "Number":
    model_choice = st.selectbox("Choose a Model", ["Neural Network", "KNN", "Logistic"], index=0)
    st.write(f"You have selected: {model_choice}")

if num_or_char == "Character":
    model_choice = st.selectbox("Choose a  model", ["character model"], index=0)
    st.write(f"You have selected: {model_choice}")

nepali_dict = {
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
    10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
    20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
    30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ'
}

st.subheader("Draw a Devanagari Character")
st.write("Reminder!!!")
st.write("Please draw in the center of the canvas below and also make sure to draw your character/number straight.")

canvas_result = st_canvas(
    stroke_width=30,
    stroke_color="white",
    background_color="black",
    width=320,
    height=320,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    img_resized = img.resize((32, 32)).convert("L")

    img_bytes = io.BytesIO()
    img_resized.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    st.download_button(
        label="Save Drawing as PNG",
        data=img_bytes,
        file_name="drawing.png",
        mime="image/png",
    )

    if st.button('Predict'):
        status_placeholder = st.empty()
        status_placeholder.write("Prediction in progress...")

        img_array = np.array(img_resized) / 255.0

        if model_choice == "character model":
            modelsel = keras.models.load_model("character.h5")
            char = modelsel.predict(img_array.reshape(1, 1024))
            prediction_result = nepali_dict[np.argmax(char)]

        elif model_choice == "Neural Network":
            modelsel = keras.models.load_model("digits.h5")
            char = modelsel.predict(img_array.reshape(1, 1024))
            prediction_result = f"Predicted digit: {np.argmax(char)}"

        elif model_choice == "KNN":
            with open("knn.pkl", "rb") as f:
                knn = pickle.load(f)
            with open("pca_250.pkl", "rb") as f:
                pca = pickle.load(f)

            img_array = img_array.reshape(1, -1)
            img_array_pca = pca.transform(img_array)
            char = knn.predict(img_array_pca)
            prediction_result = f"Prediction result: {char[0]}"

        elif model_choice == "Logistic":
            with open("logistic_regression.pkl", "rb") as f:
                logistic = pickle.load(f)
            with open("pca_250.pkl", "rb") as f:
                pca = pickle.load(f)

            img_array = img_array.reshape(1, -1)
            img_array_pca = pca.transform(img_array)
            char = logistic.predict(img_array_pca)
            prediction_result = f"Prediction result: {char[0]}"

        status_placeholder.empty()
        st.markdown(f'<h2 style="text-align: center; color: green;">{prediction_result}</h2>', unsafe_allow_html=True)
