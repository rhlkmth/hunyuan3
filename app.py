import io
import base64
import random
import requests
import os
from typing import Any, Dict, List, Tuple

import streamlit as st
from PIL import Image
import fal_client


# ---------------------------
# Page and State Configuration
# ---------------------------
st.set_page_config(page_title="Hunyuan Image 3.0 — T2I Client", layout="wide")
st.title("Hunyuan Image 3.0 — Text-to-Image Client")


# ---------------------------
# Constants and Defaults
# ---------------------------
MODEL_ID = "fal-ai/hunyuan-image/v3/text-to-image"

SAFE_PRESETS = {
    "Portrait Natural Light": "35mm portrait of a person in soft window light, shallow depth of field, subtle color grading, gentle rim light, natural skin tones",
    "Street Candid": "telephoto candid scene on a city street, dusk, neon reflections on wet pavement, dynamic composition, motion blur",
    "Product Studio": "premium studio photo of a minimalist wireless headphone on matte acrylic, softbox reflections, high key background, crisp shadows",
    "Landscape Golden Hour": "wide vista of rolling hills at golden hour, volumetric light beams, soft haze, detailed foliage, high dynamic range",
    "Concept Art Vehicle": "futuristic off-road rover concept, rugged modular design, detailed hard surface, desert environment, cinematic lighting",
    "Graphic Poster": "bold Swiss-style graphic poster, geometric shapes, strong grid, high contrast typography, limited color palette",
}

SIZE_ENUMS = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"]


# ---------------------------
# Session State
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dict: {prompt:str, results:[{data_uri, bytes, caption}], seed:int}

if "prompt_queue" not in st.session_state:
    st.session_state.prompt_queue = []  # list of str

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = SAFE_PRESETS["Portrait Natural Light"]


# ---------------------------
# Helpers
# ---------------------------
def _env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None else default


def _decode_data_uri(data_uri: str) -> bytes:
    header, b64 = data_uri.split(",", 1)
    return base64.b64decode(b64)


def _ensure_png(image_ref: str) -> Tuple[str, bytes]:
    if image_ref.startswith("data:"):
        raw = _decode_data_uri(image_ref)
    else:
        raw = requests.get(image_ref, timeout=180).content
    with Image.open(io.BytesIO(raw)) as im:
        buf = io.BytesIO()
        im.convert("RGBA").save(buf, format="PNG")
        png = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii"), png


def _subscribe(args: Dict[str, Any]) -> Dict[str, Any]:
    return fal_client.subscribe(MODEL_ID, arguments=args, with_logs=False)


# ---------------------------
# Sidebar — History
# ---------------------------
with st.sidebar:
    st.header("History")
    if not st.session_state.history:
        st.info("Generated results will appear here.")
    else:
        for idx, item in enumerate(st.session_state.history):
            st.subheader(f"Run #{len(st.session_state.history) - idx}")
            st.caption(item.get("prompt", "")[:140])
            for k, res in enumerate(item.get("results", [])):
                st.image(res["data_uri"], use_container_width=True)
                st.download_button(
                    label=f"Download #{k+1}",
                    data=res["bytes"],
                    file_name=f"run_{len(st.session_state.history)-idx}_{k+1}.png",
                    mime="image/png",
                    key=f"dl_hist_{idx}_{k}",
                    use_container_width=True,
                )
            st.divider()


# ---------------------------
# Main Layout
# ---------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("1. Build Prompt Queue")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.selectbox("Preset", list(SAFE_PRESETS.keys()), key="preset_key")
    with c2:
        if st.button("Load Preset", use_container_width=True):
            st.session_state.current_prompt = SAFE_PRESETS[st.session_state.preset_key]

    st.text_area("Prompt", key="current_prompt", height=160)

    c3, c4 = st.columns([3, 1])
    with c3:
        extra_prompts = st.text_area(
            "Bulk add (one prompt per line)", value="", placeholder="A cinematic macro photograph of a dew-covered leaf…"
        )
    with c4:
        if st.button("Add to Queue", use_container_width=True):
            items = [p.strip() for p in extra_prompts.splitlines() if p.strip()]
            if st.session_state.current_prompt.strip():
                items = [st.session_state.current_prompt.strip()] + items
            # de-dup while preserving order
            seen = set()
            merged = []
            for p in items:
                if p not in seen:
                    seen.add(p)
                    merged.append(p)
            st.session_state.prompt_queue.extend(merged)

    st.write("Queue")
    if not st.session_state.prompt_queue:
        st.info("No prompts in queue.")
    else:
        for i, p in enumerate(st.session_state.prompt_queue, 1):
            st.write(f"{i}. {p[:140]}{'…' if len(p) > 140 else ''}")
        c5, c6 = st.columns(2)
        with c5:
            if st.button("Clear Queue", use_container_width=True):
                st.session_state.prompt_queue.clear()
        with c6:
            if st.button("Pop Last", use_container_width=True):
                if st.session_state.prompt_queue:
                    st.session_state.prompt_queue.pop()

    st.divider()
    st.subheader("2. Settings")

    with st.expander("Connection"):
        fal_key = st.text_input("FAL_KEY", value=_env("FAL_KEY"), type="password", help="Never commit keys to source control.")
        if fal_key:
            os.environ["FAL_KEY"] = fal_key

    c7, c8 = st.columns(2)
    with c7:
        image_size_mode = st.selectbox("Image size", SIZE_ENUMS, index=0)
    with c8:
        output_format = st.selectbox("Output format", ["png", "jpeg"], index=0)

    width = None
    height = None
    if image_size_mode == "custom":
        c9, c10 = st.columns(2)
        with c9:
            width = st.number_input("Width", value=1280, min_value=64, max_value=4096, step=64)
        with c10:
            height = st.number_input("Height", value=1280, min_value=64, max_value=4096, step=64)

    c11, c12 = st.columns(2)
    with c11:
        num_inference_steps = st.number_input("Denoising steps", value=28, min_value=1, max_value=200, step=1)
        guidance_scale = st.number_input("Guidance scale", value=3.5, min_value=0.0, max_value=50.0, step=0.5)
        enable_safety_checker = st.checkbox("Enable safety checker", value=True)
    with c12:
        num_outputs = st.selectbox("Images per prompt", [1, 2, 3, 4], index=0)
        enable_prompt_expansion = st.checkbox("Prompt expansion", value=True)
        use_random_seed = st.checkbox("Randomize seed", value=True)
        seed_value = st.number_input("Seed", min_value=0, value=42, disabled=use_random_seed)

    st.divider()
    go = st.button(f"Generate ({len(st.session_state.prompt_queue)} in queue)", type="primary", use_container_width=True, disabled=not st.session_state.prompt_queue)

with right:
    st.subheader("Latest Results")
    if st.session_state.history:
        latest = st.session_state.history[0]
        st.caption(latest.get("prompt", "")[:200])
        for j, res in enumerate(latest.get("results", [])):
            st.image(res["data_uri"], caption=res.get("caption", ""), use_container_width=True)
            st.download_button(
                label=f"Download #{j+1}",
                data=res["bytes"],
                file_name=f"latest_{j+1}.png",
                mime="image/png",
                key=f"dl_latest_{j}",
                use_container_width=True,
            )
    else:
        st.info("No results yet.")


# ---------------------------
# Execution
# ---------------------------
if go:
    if not _env("FAL_KEY") and not fal_key:
        st.error("Missing FAL_KEY")
        st.stop()

    queue = list(st.session_state.prompt_queue)
    st.session_state.prompt_queue.clear()

    status = st.status("Starting generation…", expanded=True)
    run_history_batch: List[Dict[str, Any]] = []

    for idx, prompt in enumerate(queue, 1):
        status.update(label=f"Generating {idx}/{len(queue)}")

        args: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": "",
            "num_images": int(num_outputs),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": output_format,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "sync_mode": True,  # return data URIs
        }

        if image_size_mode == "custom":
            args["image_size"] = {"width": int(width or 0), "height": int(height or 0)}
        else:
            args["image_size"] = image_size_mode

        if not use_random_seed:
            args["seed"] = int(seed_value)
        else:
            args["seed"] = random.randint(0, 2**31 - 1)

        try:
            result = _subscribe(args)
        except Exception as e:
            st.error(f"Generation failed for item {idx}: {e}")
            continue

        images = result.get("images") or []
        base_seed = result.get("seed")
        batch = {"prompt": prompt, "seed": base_seed, "results": []}

        for k, entry in enumerate(images):
            url = entry.get("url", "")
            data_uri, png_bytes = _ensure_png(url)
            caption = f"Output #{k+1} • Seed: {base_seed}"
            batch["results"].append({"data_uri": data_uri, "bytes": png_bytes, "caption": caption})

        if batch["results"]:
            run_history_batch.insert(0, batch)

    if run_history_batch:
        st.session_state.history = run_history_batch + st.session_state.history

    status.update(label="Generation complete", state="complete")
