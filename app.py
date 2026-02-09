import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import requests
from io import BytesIO

st.set_page_config(page_title="Caption Generator", page_icon="🖼️")

st.title("🖼️ Caption Generator")
st.write("Upload an image URL and get an automatic AI caption.")

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

image_url = st.text_input("Enter Image URL")

if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        st.image(image, caption="Loaded Image", width="stretch")

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                pixel_values = processor(images=image, return_tensors="pt").pixel_values
                output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.success("Caption Generated!")
            st.write(caption)

    except:
        st.error("Invalid image URL. Try another.")
