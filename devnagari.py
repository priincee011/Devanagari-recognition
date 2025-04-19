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
else:
    model_choice = st.selectbox("Choose a model", ["character model"], index=0)

st.write(f"You have selected: {model_choice}")

nepali_dict = {
    0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
    10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
    20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
    30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ'
}
devanagari_digits = {
    0: '०', 1: '१', 2: '२', 3: '३', 4: '४',
    5: '५', 6: '६', 7: '७', 8: '८', 9: '९'
}

st.subheader("Draw a Devanagari Character")
st.write("Draw in the center and keep your stroke straight.")
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
    buf = io.BytesIO()
    img_resized.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Save Drawing as PNG", data=buf, file_name="drawing.png", mime="image/png")

    if st.button("Predict"):
        status = st.empty()
        status.write("Prediction in progress...")

        x = np.array(img_resized).reshape(1, -1) / 255.0

        if model_choice == "character model":
            model = keras.models.load_model("character.h5")
            probs = model.predict(x)
            pred = np.argmax(probs)
            result = f"Predicted character: {nepali_dict[pred]}"
            st.write("Top 5 Predictions:")
            flat = probs.flatten()
            for idx in np.argsort(flat)[-5:][::-1]:
                if idx in nepali_dict:
                    st.write(f"{nepali_dict[idx]}: {flat[idx]*100:.3f}%")

        elif model_choice == "Neural Network":
            model = keras.models.load_model("digits.h5")
            probs = model.predict(x)
            pred = np.argmax(probs)
            result = f"Predicted digit: {pred} ({devanagari_digits[pred]})"
            st.write("Top 5 Predictions:")
            flat = probs.flatten()
            for idx in np.argsort(flat)[-5:][::-1]:
                if idx in devanagari_digits:
                    st.write(f"{idx} ({devanagari_digits[idx]}): {flat[idx]*100:.3f}%")

        elif model_choice == "KNN":
            with open("knn.pkl", "rb") as f:
                knn = pickle.load(f)
            with open("pca_250.pkl", "rb") as f:
                pca = pickle.load(f)
            x_pca = pca.transform(x)
            pred = knn.predict(x_pca)[0]
            result = f"Predicted digit: {pred} ({devanagari_digits[int(pred)]})"
            if hasattr(knn, "predict_proba"):
                probs = knn.predict_proba(x_pca)[0]
                st.write("Top 5 Predictions:")
                flat = np.array(probs).flatten()
                for idx in np.argsort(flat)[-5:][::-1]:
                    if idx in devanagari_digits:
                        st.write(f"{idx} ({devanagari_digits[idx]}): {flat[idx]*100:.3f}%")

        else:
            with open("logistic_regression.pkl", "rb") as f:
                logreg = pickle.load(f)
            with open("pca_250.pkl", "rb") as f:
                pca = pickle.load(f)
            x_pca = pca.transform(x)
            pred = logreg.predict(x_pca)[0]
            result = f"Predicted digit: {pred} ({devanagari_digits[int(pred)]})"
            if hasattr(logreg, "predict_proba"):
                probs = logreg.predict_proba(x_pca)[0]
                st.write("Top 5 Predictions:")
                flat = np.array(probs).flatten()
                for idx in np.argsort(flat)[-5:][::-1]:
                    if idx in devanagari_digits:
                        st.write(f"{idx} ({devanagari_digits[idx]}): {flat[idx]*100:.3f}%")

        status.empty()
        st.markdown(f'<h2 style="text-align:center;color:green;">{result}</h2>', unsafe_allow_html=True)
