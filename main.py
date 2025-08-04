import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()


# Set Streamlit page config
st.set_page_config(page_title="Text-to-Image | FLUX.1", layout="centered")
st.title("🖼️ Text-to-Image using FLUX.1 (Nebiuss)")
st.caption("Enter a prompt and generate an image using the FLUX.1 model.")

# Input: prompt
prompt = st.text_input("Enter your prompt:", "Astronaut riding a horse")

# Button to generate
if st.button("Generate Image"):
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN:
        st.error("⚠️ Hugging Face API token not found. Please set HF_TOKEN as an environment variable.")
        st.stop()

    try:
        # Initialize the inference client
        client = InferenceClient(
            provider="nebius",
            api_key=HF_TOKEN,
        )

        with st.spinner("Generating image..."):
            # Generate image from text
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev"
            )

        # Display the result
        st.image(image, caption="Generated Image", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")


