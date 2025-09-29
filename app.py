# app.py
import io
import base64
import requests
import random
import streamlit as st
from PIL import Image

# ---------------------------
# Page and State Configuration
# ---------------------------
st.set_page_config(
    page_title="Hunyuan AI Image Generator",
    layout="wide",  # Use wide layout for better result display
    page_icon="🎨"
)

# ---------------------------
# API Key & Constants
# ---------------------------
# IMPORTANT: Replace with your actual Fal.ai API key
YOUR_FAL_API_KEY = "8055255a-ec46-48ed-96d7-df463cb6fbd6:29917914048fdc23abffd017879f475a"
FAL_ENDPOINT = "https://fal.run/fal-ai/hunyuan-image/v3/text-to-image"

# SFW, high-quality prompts for style inspiration
PREDEFINED_PROMPTS = {
    "Cinematic Sunset": "A detailed cinematic photo of a beautiful golden retriever wearing a small crown, standing in a magical forest at sunset, highly detailed, fantasy art, volumetric light.",
    "Cyberpunk Rain": "An elegant watercolor painting of a bustling cyberpunk city street on a rainy night, neon signs reflecting in puddles, intricate details, moody lighting.",
    "Space Cat": "A photorealistic image of a cat in an astronaut suit floating in space, looking at Earth, 8k, dramatic lighting, detailed textures.",
    "Abstract Geometry": "Abstract 3D render of interconnected geometric shapes in pastel colors, soft studio lighting, high resolution, minimalist.",
    "Viking Ship": "An epic matte painting of a Viking longship sailing through a stormy sea towards a hidden coastline, digital art, dramatic clouds, hyper-detailed.",
    "Steampunk Robot": "A vintage sepia photograph of a finely detailed Steampunk-style robot sitting at a desk and writing a letter, intricate brass mechanics, soft lighting.",
    "Underwater Garden": "A vibrant, luminescent underwater garden with exotic glowing plants and bioluminescent fish, macro lens, deep sea environment.",
    "Fairy Tale Village": "A whimsical, illustrated village nestled in the hollow of a giant ancient tree, clear sunny day, storybook style, high saturation.",
}

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, watermark, signature, deformed, worst quality, jpeg artifacts, poorly drawn, text, logo"
IMAGE_SIZES = ["square_hd", "portrait_hd", "landscape_hd"]

# ---------------------------
# Session State Initialization
# ---------------------------
def initialize_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = PREDEFINED_PROMPTS["Cinematic Sunset"]
    if 'image_size' not in st.session_state:
        st.session_state.image_size = "square_hd"
    if 'negative_prompt' not in st.session_state:
        st.session_state.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    if 'num_outputs' not in st.session_state:
        st.session_state.num_outputs = "1"
    if 'use_random_seed' not in st.session_state:
        st.session_state.use_random_seed = True
    if 'guidance_scale' not in st.session_state:
        st.session_state.guidance_scale = 3.5
    if 'num_inference_steps' not in st.session_state:
        st.session_state.num_inference_steps = 28
    if 'api_key' not in st.session_state:
        st.session_state.api_key = YOUR_FAL_API_KEY

initialize_state()

# ---------------------------
# API Helper Function
# ---------------------------
def call_fal_generate(api_key, **kwargs):
    """Calls the Hunyuan Text-to-Image API with dynamic arguments."""
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    
    # Base payload
    payload = {
        "sync_mode": True,
        "output_format": "png",
        "enable_safety_checker": False, # As requested
    }
    # Add all other arguments passed to the function
    payload.update(kwargs)

    resp = requests.post(FAL_ENDPOINT, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("images"):
        raise RuntimeError("The API returned no images.")
    return data

def ensure_png_output(image_reference: str):
    """Fetches image from URL and ensures it is returned as a data URI and PNG bytes."""
    image_bytes = base64.b64decode(image_reference.split(",", 1)[1]) if image_reference.startswith("data:") else requests.get(image_reference, timeout=180).content
    with Image.open(io.BytesIO(image_bytes)) as img:
        buffer = io.BytesIO()
        img.convert("RGBA").save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii"), png_bytes

# ---------------------------
# UI: Sidebar (Control Panel)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.write("Configure your image generation settings here.")
    st.divider()

    st.subheader("🎨 Style & Prompt")
    
    def update_prompt_from_style():
        selected_style = st.session_state.prompt_style_key
        st.session_state.current_prompt = PREDEFINED_PROMPTS.get(selected_style, "")

    st.selectbox(
        "Style Presets", 
        list(PREDEFINED_PROMPTS.keys()), 
        key="prompt_style_key", 
        on_change=update_prompt_from_style,
        help="Select a preset to fill the prompt with a sample style."
    )
    
    st.text_area(
        "Negative Prompt",
        key="negative_prompt",
        height=100,
        help="Describe what you DON'T want in the image."
    )

    st.divider()
    st.subheader("🔧 Generation Settings")
    
    st.radio(
        "Image Size", 
        IMAGE_SIZES, 
        key="image_size", 
        horizontal=True, 
        help="Aspect ratio of the output image."
    )
    
    st.radio(
        "Number of Outputs", 
        ["1", "2", "3", "4"], 
        key="num_outputs", 
        horizontal=True, 
        help="Number of images to generate at once."
    )
    
    # Seed controls
    num_outputs = int(st.session_state.num_outputs)
    force_random = num_outputs > 1
    use_random_seed = st.checkbox(
        "Randomize Seed", 
        value=True, 
        disabled=force_random, 
        key="use_random_seed", 
        help="Use a random seed for each generation. (Forced for multiple outputs)"
    )
    if not use_random_seed:
        st.number_input("Seed", min_value=0, value=42, key="seed")

    with st.expander("Advanced Settings"):
        st.slider(
            "Guidance Scale (CFG)", 
            min_value=1.0, 
            max_value=20.0, 
            value=3.5, 
            step=0.5, 
            key="guidance_scale",
            help="How strictly the model follows the prompt. Higher values are more strict."
        )
        st.slider(
            "Inference Steps", 
            min_value=1, 
            max_value=50, 
            value=28, 
            key="num_inference_steps",
            help="More steps can improve quality but take longer."
        )
        st.text_input("Fal.ai API Key", value=st.session_state.api_key, key="api_key", type="password")

# ---------------------------
# UI: Main Page
# ---------------------------
st.title("🎨 Hunyuan AI Image Generator")
st.write("Craft your vision. Describe anything you can imagine and see it come to life.")

# Prompt input area
st.text_area(
    "Enter your prompt here:",
    key="current_prompt",
    height=150,
    label_visibility="collapsed"
)

# Generate button
if st.button("🚀 Generate Image", use_container_width=True, type="primary"):
    with st.spinner("Calling the AI artist... a masterpiece is on its way!"):
        try:
            # Prepare API arguments
            api_args = {
                "prompt": st.session_state.current_prompt,
                "negative_prompt": st.session_state.negative_prompt,
                "image_size": st.session_state.image_size,
                "num_images": int(st.session_state.num_outputs),
                "guidance_scale": st.session_state.guidance_scale,
                "num_inference_steps": st.session_state.num_inference_steps,
            }
            
            # Handle seed
            if not st.session_state.use_random_seed and not force_random:
                api_args["seed"] = st.session_state.seed

            # Make the API call
            response_data = call_fal_generate(st.session_state.api_key, **api_args)
            
            # Process and store results in history
            generation_batch = {"results": []}
            images_list = response_data["images"]
            base_seed = response_data.get("seed", "N/A")
            
            for j, img_data in enumerate(images_list):
                png_data_uri, png_bytes = ensure_png_output(img_data["url"])
                caption = f"Output #{j+1} | Seed: {base_seed}"
                generation_batch["results"].append({"data_uri": png_data_uri, "bytes": png_bytes, "caption": caption})

            if generation_batch["results"]:
                st.session_state.history.insert(0, generation_batch)
            
            st.rerun()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("API Error (401): The API key is invalid or expired. Please check it in the sidebar.")
            else:
                st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.divider()

# Display the latest result
if st.session_state.history:
    st.subheader("✨ Latest Result")
    latest_generation = st.session_state.history[0]
    results_to_display = latest_generation.get('results', [])
    
    cols = st.columns(len(results_to_display))
    for i, res in enumerate(results_to_display):
        with cols[i]:
            st.image(res["data_uri"], caption=res.get("caption", ""))
            st.download_button(
                label=f"Download #{i+1}",
                data=res["bytes"],
                file_name=f"generated_image_{i+1}.png",
                mime="image/png",
                key=f"latest_dl_{i}",
                use_container_width=True
            )
else:
    st.info("Your generated images will appear here.")

# Display session history in an expander
with st.expander("📜 View Session History"):
    if not st.session_state.history:
        st.write("No generations yet. Your history will be stored here.")
    else:
        # Skip the latest generation since it's already displayed above
        for i, past_gen in enumerate(st.session_state.history[1:]):
            st.subheader(f"Past Generation #{i+1}")
            results = past_gen.get('results', [])
            
            # Create columns for each image in the past generation
            cols = st.columns(len(results))
            for idx, res in enumerate(results):
                with cols[idx]:
                    st.image(res["data_uri"], caption=res.get("caption", ""))
            st.divider()
