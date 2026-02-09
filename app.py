import streamlit as st
from PIL import Image
from transformers import pipeline
import requests
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# ---------------- CUSTOM CSS (Better UI) ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}
.sub-text {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.caption-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🖼️ AI Image Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Paste an image URL and generate an automatic AI caption</div>', unsafe_allow_html=True)

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-text-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

captioner = load_model()

# ---------------- INPUT SECTION ----------------
image_url = st.text_input("🔗 Enter Image URL")

# ---------------- PROCESS ----------------
if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("✨ Generate Caption"):
            with st.spinner("Generating caption..."):
                result = captioner(image)[0]["generated_text"]

            st.markdown(
                f'<div class="caption-box">📜 {result}</div>',
                unsafe_allow_html=True
            )

    except:
        st.error("❌ Invalid image URL. Please try another image link.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers")
