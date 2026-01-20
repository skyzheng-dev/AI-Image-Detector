import cv2
import numpy as np
import streamlit as st

# AI image detector which is already trained
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


# loading AI image detector
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

# convert the image into the correct format for AI to detect
def preprocess_image(image):
    # making the img an array of numbers each representing the pixels
    img = np.array(image)
    # resize the img
    img = cv2.resize(img,(224,224))
    # preprocess the input (img) so it can be later sent to the MobileNetV2 AI
    img = preprocess_input(img)
    # creating another dimension for the img, i.e encompassing the img array within a list
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]   # [0.9,0.5,0.2,0.0] grabbing the top three most confident predictions
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🏞️", layout="centered")

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")
    
    # caching the model
    @st.cache_resource
    # load the model so you don't have to keep reloading the page for new changes
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type= ["jpg","png"])

    # if there is an uploaded img, display it and a button
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", width= 700
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

# if running the python file directly, run the main function
if __name__ == "__main__":
    main()