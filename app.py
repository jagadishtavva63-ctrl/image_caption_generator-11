import streamlit as st
from PIL import Image
from transformers import pipeline
import requests
from io import BytesIO

st.set_page_config(page_title="Caption Generator", page_icon="🖼️")

st.title("🖼️ Caption Generator")
st.write("Paste an image URL and get an automatic AI caption.")

@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

captioner = load_model()

image_url = st.text_input("Enter Image URL")

if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        st.image(image, caption="Loaded Image", width="stretch")

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                result = captioner(image)[0]["generated_text"]

            st.success("Caption Generated!")
            st.write(result)

    except:
        st.error("Invalid image URL. Try another.")
