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

# Predefined Prompts - Edited to be suitable for Text-to-Image generation
PREDEFINED_PROMPTS = {
    "bedroom": "nude, sexy model, soft natural round tits, medium brown nipples, oiled wet body, bright warm lighting, professional sony alpha photo quality,wearing black g string, large round saggy natural breasts, background sensual bedroom dark",
    " bedroom 2": "nude, sexy model, soft natural round tits, medium brown nipples, oiled wet body, bright warm lighting, professional sony alpha photo quality,nude vagina, trimmed pubic hair, large round saggy natural breasts, background sensual bedroom dark, sensual nudelarge saggy boobs, hard nipples, erect nipples",
    "oily": "sensual nude, with naked breasts or butts ass visible, depending on the angle, small or large breasts depending on the image, naked breasts visible, shiny",
    "NUDE stadium": "sensual nude, with naked breasts or butts ass visible, depending on the angle, small or large breasts depending on the image, naked breasts visible, shiny, large stadium dark background",
    "Lorena style": "nude, sexy model, soft natural round tits, medium brown nipples, oiled wet body, bright warm lighting, professional sony alpha photo quality,wearing black g string",
    "Poolside Shine": "sensual nude, naked ass visible from rear angle, medium natural breasts, wet and glossy skin, bright poolside sunlight, high res, background luxurious infinity pool overlooking ocean",
    "Mountain Breeze": "nude, soft sagging tits, medium tan nipples, lightly oiled body, cool mountainous dawn light, professional sony alpha photo quality, wearing black bikini bottom, large natural breasts, background rugged mountain peaks with fog",
    "Desert Heat": "sensual nude, breasts visible in profile, small firm boobs, sandy wet sheen, harsh desert sun lighting, high res, background vast sand dunes under blue sky",
    "Studio Elegance": "nude, sexy model, rounded perky tits, light brown nipples, oiled silky body, soft studio warm light, professional fuji photo quality, wearing sheer thong, medium breasts, background minimalist white studio with drapes",
    "Rainforest Dew": "sensual nude, naked breasts and ass visible depending on angle, large natural breasts, dewy wet skin, lush green tropical light, high res, background dense rainforest with waterfalls",
    "City Rooftop": "nude, firm natural boobs, pinkish nipples, glistening oiled body, twilight city skyline lighting, professional canon photo quality, wearing lace g-string, small perky breasts, background high-rise rooftop with city lights below",
    "Snowy Cabin": "sensual nude, side view of naked breasts, medium saggy breasts, frosty wet glow, warm indoor firelight, high res, background cozy wooden cabin in snowy woods",
    "Garden Bloom": "nude, sexy model, soft round tits, dark nipples, oiled floral-scented body, bright garden sunlight, professional nikon photo quality, wearing floral thong, large natural breasts, background blooming flower garden with vines",
    "Cave Mystery": "sensual nude, naked ass from back angle, small natural boobs, damp shiny skin, dim torchlight in cave, high res, background mysterious underground cave with stalactites",
    "Luxury Yacht": "nude, perky sagging tits, light tan nipples, wet oiled body from sea spray, nautical sunset lighting, professional sony alpha photo quality, wearing nautical g-string, medium breasts, background deck of a luxury yacht at sea",
    "Art Gallery": "sensual nude, breasts visible frontally, large firm breasts, glossy wet sheen, elegant gallery spotlights, high res, background modern art gallery with abstract paintings",
    "Volcanic Steam": "nude, sexy model, round natural boobs, brown nipples, steamy oiled body, red volcanic glow lighting, professional fuji photo quality, wearing black thong, small soft breasts, background steaming volcanic landscape with lava flows",
    "Lakeside Dawn": "sensual nude, naked profile with breasts and hips, medium natural tits, misty wet skin, soft dawn light over water, high res, background serene lake with misty mountains",
    "Neon Club": "nude, full perky boobs, pink nipples, shiny oiled body, pulsing neon club lights, professional canon photo quality, wearing glow-in-dark g-string, large breasts, background vibrant nightclub with dance floor",
    "Ancient Ruins": "sensual nude, ass visible from low angle, small saggy breasts, dusty wet glow, golden ancient sunlight, high res, background crumbling ancient ruins with ivy",
    "Spa Retreat": "nude, sexy model, soft firm tits, light brown nipples, oiled relaxed body, warm spa ambient lighting, professional nikon photo quality, wearing towel thong, medium natural breasts, background luxurious spa with steam rooms",
    "Desert Oasis": "sensual nude, naked breasts in three-quarter view, large perky boobs, watery wet skin, oasis palm shade lighting, high res, background palm-fringed desert oasis with pool",
    "Skyline Penthouse": "nude, rounded sagging tits, tan nipples, glistening oiled body, city night view lighting, professional sony alpha photo quality, wearing silk g-string, small breasts, background penthouse balcony overlooking skyline",
    "Tropical Hut": "sensual nude, side ass and breasts visible, medium firm breasts, humid wet sheen, thatched hut lantern light, high res, background bamboo tropical hut on stilts",
    "Ice Cave": "nude, sexy model, perky natural boobs, light pink nipples, icy oiled body, blue glacial glow lighting, professional fuji photo quality, wearing fur-trimmed thong, large saggy breasts, background shimmering ice cave with aurora",
    "Vineyard Sunset": "sensual nude, naked from behind with breasts hinted, small natural tits, vine-dewed wet skin, warm vineyard sunset, high res, background rolling vineyard hills with grapevines",
    "Futuristic Lab": "nude, full round tits, dark nipples, shiny synthetic oil body, holographic lab lighting, professional canon photo quality, wearing metallic g-string, medium breasts, background high-tech futuristic lab with screens",
    "Autumn Woods": "sensual nude, breasts visible in motion, large soft boobs, leafy wet glow, golden autumn foliage light, high res, background colorful autumn woods with falling leaves",
    "Beach Cave": "nude, sexy model, soft perky tits, brown nipples, salty oiled body, cave-filtered sunlight, professional nikon photo quality, wearing seashell thong, small natural breasts, background hidden beach cave with ocean view",
    "Midnight Garden": "sensual nude, naked ass and side breasts, medium saggy tits, moonlit wet skin, silvery midnight garden light, high res, background enchanted midnight garden with fireflies",
    "Industrial Loft": "nude, firm natural boobs, pinkish nipples, greased oiled body, exposed brick industrial lighting, professional sony alpha photo quality, wearing chain g-string, large breasts, background urban industrial loft with exposed pipes",
    "Coral Reef": "sensual nude, underwater breasts visible, small firm tits, bubbly wet sheen, turquoise reef sunlight filtering, high res, background vibrant coral reef with tropical fish",
}

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = PREDEFINED_PROMPTS["bedroom"]
# New state for image size
if 'image_size' not in st.session_state:
    st.session_state.image_size = "square_hd"

# ---------------------------
# Helper Functions
# ---------------------------
# Removed: upload_local_image
# Removed: compute_scaled_size

def call_fal_generate(api_key: str, image_size: str, prompt: str, seed: int = None, num_images: int = 1):
    """Calls the Hunyuan Text-to-Image API."""
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "image_size": image_size,
        "num_images": num_images,
        "enable_safety_checker": False, # Retained from original code
        "sync_mode": True,
        "output_format": "png", # Ensure consistent output format
    }
    if seed is not None:
        payload["seed"] = seed
        
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
            elif isinstance(past_gen, list):
                st.info("Original context not available for this older session.")
                results_to_display = past_gen
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
    
    # Removed: File uploader and Image Queue section

    if st.button("🚀 Generate Image", key="generate_button", use_container_width=True):
        try:
            processing_status = st.status("Starting image generation...", expanded=True)
            
            # Get settings from state
            image_size = st.session_state.get("image_size", "square_hd")
            num_outputs = int(st.session_state.get("num_outputs", 1))
            prompt = st.session_state.current_prompt
            
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
                    current_seed, 
                    num_outputs
                )
            
            # History item structure for T2I
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
            st.error(f"API Error (401): The API key is likely invalid or expired." if hasattr(e, 'response') and e.response.status_code == 401 else f"API Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    st.divider()
    st.subheader("2. Configure Settings")
    control_col1, control_col2 = st.columns(2)
    
    IMAGE_SIZES = ["square_hd", "portrait_hd", "landscape_hd"] # Options from docs
    with control_col1:
        # Replaced Resolution Upscale with Image Size
        st.radio("Image Size", IMAGE_SIZES, index=0, key="image_size", horizontal=True, help="Aspect ratio and resolution for the generated image.")
        num_outputs_str = st.radio("Number of Outputs", ["1", "2", "3", "4"], index=0, key="num_outputs", horizontal=True, help="Number of images to generate per prompt.")
        num_outputs = int(num_outputs_str)
    
    with control_col2:
        st.selectbox("Prompt Style", list(PREDEFINED_PROMPTS.keys()), key="prompt_style_key", on_change=update_prompt_from_style)
        force_random = num_outputs > 1
        use_random_seed = st.checkbox("Randomize Seed", value=True, disabled=force_random, key="use_random_seed", help="Use a random seed for each generation. Required for multiple outputs.")
        if not use_random_seed:
            seed = st.number_input("Seed", min_value=0, value=42, key="seed")

    st.write("Prompt (Text-to-Image)")
    prompt_text = st.text_area("Prompt", height=150, value=st.session_state.current_prompt, key="current_prompt_text", label_visibility="collapsed")
    if prompt_text != st.session_state.current_prompt:
        st.session_state.current_prompt = prompt_text
    
    with st.expander("Advanced Settings"):
        api_key_to_use = st.text_input("Enter a different API Key (optional)", value=YOUR_FAL_API_KEY, key="api_key")
        
with right_col:
    st.subheader("Latest Result")
    if st.session_state.history:
        latest_generation = st.session_state.history[0]
        
        results_to_display = latest_generation.get('results', [])
        
        if results_to_display:
            st.write("**Generated Outputs**")
        
        # Removed logic to display 'original' image
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
