# app.py
import io
import base64
import requests
import random
import streamlit as st
from PIL import Image
# from fal_client import SyncClient # Not needed for T2I via HTTP POST

# ---------------------------
# Page and State Configuration
# ---------------------------
st.set_page_config(page_title="Hunyuan AI Image Generator", layout="centered", page_icon="🖼️")
st.title("Hunyuan AI Image Generator")

# ---------------------------
# API Key & Constants
# ---------------------------
YOUR_FAL_API_KEY = "8055255a-ec46-48ed-96d7-df463cb6fbd6:29917914048fdc23abffd017879f475a"

# Updated FAL Endpoint for Hunyuan Text-to-Image
FAL_ENDPOINT = "https://fal.run/fal-ai/hunyuan-image/v3/text-to-image"

# **FIX:** Replaced NSFW prompts with safe, high-quality T2I prompts to prevent 422 error.
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

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, watermark, signature, deformed, worst quality, jpeg artifacts, poorly drawn"
IMAGE_SIZES = ["square_hd", "portrait_hd", "landscape_hd"] # Options from docs

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = PREDEFINED_PROMPTS["Cinematic Sunset"]
if 'image_size' not in st.session_state:
    st.session_state.image_size = "square_hd"
if 'negative_prompt' not in st.session_state:
    st.session_state.negative_prompt = DEFAULT_NEGATIVE_PROMPT
if 'num_outputs' not in st.session_state:
    st.session_state.num_outputs = 1
if 'use_random_seed' not in st.session_state:
    st.session_state.use_random_seed = True

# ---------------------------
# Helper Functions
# ---------------------------
def call_fal_generate(api_key: str, image_size: str, prompt: str, negative_prompt: str, seed: int = None, num_images: int = 1):
    """Calls the Hunyuan Text-to-Image API."""
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt, # **FIX:** Added negative_prompt
        "image_size": image_size,
        "num_images": num_images,
        # **FIX:** Removed 'enable_safety_checker': False to rely on API default, as it was likely causing issues with explicit prompts.
        "sync_mode": True,
        "output_format": "png", 
    }
    if seed is not None:
        payload["seed"] = seed
        
    resp = requests.post(FAL_ENDPOINT, headers=headers, json=payload, timeout=180)
    resp.raise_for_status() # Will raise HTTPError on 422, which is handled below
    data = resp.json()
    if not data.get("images"):
        raise RuntimeError("The API returned no images.")
    return data

def ensure_png_output(image_reference: str):
    """Fetches image from URL and ensures it is returned as a data URI and PNG bytes."""
    # This logic is retained from the original code for handling the API response format
    image_bytes = base64.b64decode(image_reference.split(",", 1)[1]) if image_reference.startswith("data:") else requests.get(image_reference, timeout=180).content
    with Image.open(io.BytesIO(image_bytes)) as img:
        buffer = io.BytesIO()
        img.convert("RGBA").save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii"), png_bytes

def update_prompt_from_style():
    """Updates the prompt from the selected style in the selectbox."""
    selected_style = st.session_state.prompt_style_key
    if selected_style in PREDEFINED_PROMPTS:
        st.session_state.current_prompt = PREDEFINED_PROMPTS[selected_style]

# ---------------------------
# UI Layout
# ---------------------------
with st.sidebar:
    st.header("Session History")
    if not st.session_state.history:
        st.info("Your previous generations will be stored here.")
    else:
        for i, past_gen in enumerate(st.session_state.history):
            st.subheader(f"Session #{len(st.session_state.history) - i}")

            # Handle current and potential old list format
            if isinstance(past_gen, dict) and 'results' in past_gen:
                results_to_display = past_gen.get('results', [])
            else:
                continue

            for idx, res in enumerate(results_to_display):
                if isinstance(res, dict) and "data_uri" in res:
                    st.image(res["data_uri"], caption=res.get("caption", ""))
                    st.download_button(
                        label="Download",
                        data=res["bytes"],
                        file_name=f"history_{i}_{idx}.png",
                        mime="image/png",
                        key=f"history_dl_{i}_{idx}"
                    )
            st.divider()

main_col, right_col = st.columns([1, 1])

with main_col:
    st.subheader("1. Generate Image")
    
    if st.button("🚀 Generate Image", key="generate_button", use_container_width=True):
        try:
            processing_status = st.status("Starting image generation...", expanded=True)
            
            # Get settings from state
            image_size = st.session_state.get("image_size", "square_hd")
            num_outputs = int(st.session_state.get("num_outputs", 1))
            prompt = st.session_state.current_prompt
            negative_prompt = st.session_state.negative_prompt
            
            force_random = num_outputs > 1
            
            current_seed = None
            if st.session_state.get("use_random_seed", True) or force_random:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                current_seed = st.session_state.get("seed", 42)

            processing_status.update(label=f"Generating {num_outputs} image(s) with size {image_size} and seed {current_seed}...")

            with st.spinner(f"Generating {num_outputs} output(s)..."):
                response_data = call_fal_generate(
                    st.session_state.get("api_key", YOUR_FAL_API_KEY), 
                    image_size,
                    prompt, 
                    negative_prompt, # **FIX:** Pass negative_prompt
                    current_seed, 
                    num_outputs
                )
            
            generation_batch = {"results": []} 
            images_list = response_data["images"]
            base_seed = response_data["seed"]
            
            for j, img_data in enumerate(images_list):
                png_data_uri, png_bytes = ensure_png_output(img_data["url"])
                caption = f"Output #{j+1} • Seed: {base_seed} • Size: {image_size}"
                generation_batch["results"].append({"data_uri": png_data_uri, "bytes": png_bytes, "caption": caption})

            if generation_batch["results"]:
                st.session_state.history.insert(0, generation_batch)

            processing_status.update(label="✅ Image generation complete!", state="complete")
            st.rerun()

        except requests.exceptions.HTTPError as e:
            # Improved error message for common 422 cause
            if hasattr(e, 'response') and e.response.status_code == 422:
                st.error(f"API Error (422 Client Error): The prompt may be violating the model's content policy, or an input parameter is invalid. Please check your prompt and settings.")
            elif hasattr(e, 'response') and e.response.status_code == 401:
                st.error(f"API Error (401): The API key is likely invalid or expired.")
            else:
                st.error(f"API Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.divider()
    st.subheader("2. Configure Settings")
    
    # Text Area for Prompt
    st.write("Prompt (Text-to-Image)")
    prompt_text = st.text_area("Prompt", height=150, value=st.session_state.current_prompt, key="current_prompt_text", label_visibility="collapsed")
    if prompt_text != st.session_state.current_prompt:
        st.session_state.current_prompt = prompt_text

    # Text Area for Negative Prompt
    st.write("Negative Prompt (Guide generation away from these)")
    negative_prompt_text = st.text_area("Negative Prompt", height=75, value=st.session_state.negative_prompt, key="negative_prompt_text", label_visibility="collapsed")
    if negative_prompt_text != st.session_state.negative_prompt:
        st.session_state.negative_prompt = negative_prompt_text

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        st.radio("Image Size", IMAGE_SIZES, index=0, key="image_size", horizontal=True, help="Aspect ratio and resolution for the generated image.")
        num_outputs_str = st.radio("Number of Outputs", ["1", "2", "3", "4"], index=0, key="num_outputs", horizontal=True, help="Number of images to generate per prompt.")
        num_outputs = int(num_outputs_str)
    
    with control_col2:
        st.selectbox("Prompt Style", list(PREDEFINED_PROMPTS.keys()), key="prompt_style_key", on_change=update_prompt_from_style)
        force_random = num_outputs > 1
        use_random_seed = st.checkbox("Randomize Seed", value=st.session_state.use_random_seed, disabled=force_random, key="use_random_seed", help="Use a random seed for each generation. Required for multiple outputs.")
        if not use_random_seed:
            seed = st.number_input("Seed", min_value=0, value=42, key="seed")

    
    with st.expander("Advanced Settings"):
        api_key_to_use = st.text_input("Enter a different API Key (optional)", value=YOUR_FAL_API_KEY, key="api_key")
        
with right_col:
    st.subheader("Latest Result")
    if st.session_state.history:
        latest_generation = st.session_state.history[0]
        
        results_to_display = latest_generation.get('results', [])
        
        if results_to_display:
            st.write("**Generated Outputs**")
        
        for idx, res in enumerate(results_to_display):
            if isinstance(res, dict) and "data_uri" in res:
                st.image(res["data_uri"], caption=res.get("caption", ""))
                st.download_button(
                    label=f"Download Output #{idx+1}",
                    data=res["bytes"],
                    file_name=f"latest_{idx}.png",
                    mime="image/png",
                    key=f"latest_dl_{idx}"
                )
    else:
        st.info("Your latest generated images will appear here.")
