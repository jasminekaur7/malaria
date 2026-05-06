import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import gdown

import tensorflow as tf
from tensorflow.keras import layers, models

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html, body, .stApp { background-color: #0a0a0f !important; color: #e8e8f0 !important; font-family: 'Syne', sans-serif; }
.hero { background: linear-gradient(135deg, #0a0a0f 0%, #0d1a12 50%, #0a0a0f 100%); border-bottom: 1px solid #1e1e2e; padding: 2.5rem 3rem 2rem; margin-bottom: 2rem; }
.hero-tag { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00ff88; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.5rem; }
.hero-title { font-size: 2.8rem; font-weight: 800; line-height: 1.1; margin: 0; background: linear-gradient(90deg, #fff 0%, #00ff88 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { color: #6b6b8a; font-family: 'Space Mono', monospace; font-size: 0.78rem; margin-top: 0.6rem; }
.card { background: #12121a; border: 1px solid #1e1e2e; border-radius: 12px; padding: 1.8rem; margin-bottom: 1.2rem; }
.card-title { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #00ff88; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 1rem; padding-bottom: 0.6rem; border-bottom: 1px solid #1e1e2e; }
.result-infected { background: linear-gradient(135deg, #1a0a0e, #2a0f18); border: 1px solid #ff4466; border-radius: 12px; padding: 2rem; text-align: center; }
.result-healthy { background: linear-gradient(135deg, #0a1a0e, #0f2a18); border: 1px solid #00ff88; border-radius: 12px; padding: 2rem; text-align: center; }
.result-label { font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; }
.result-conf { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #6b6b8a; letter-spacing: 0.1em; }
.stat-box { background: #0d0d14; border: 1px solid #1e1e2e; border-radius: 8px; padding: 1rem; text-align: center; }
.stat-val { font-size: 1.2rem; font-weight: 800; color: #00ff88; font-family: 'Space Mono', monospace; }
.stat-label { font-size: 0.65rem; color: #6b6b8a; font-family: 'Space Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.2rem; }
.step { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; }
.step-num { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00ff88; background: #0d1a12; border: 1px solid #00ff88; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }
.step-text { font-size: 0.85rem; color: #6b6b8a; line-height: 1.5; }
.disclaimer { background: #0f0f18; border: 1px solid #1e1e2e; border-left: 3px solid #ffaa00; border-radius: 8px; padding: 1rem 1.2rem; font-family: 'Space Mono', monospace; font-size: 0.65rem; color: #6b6b8a; line-height: 1.6; }
.stButton > button { background: #00ff88 !important; color: #000 !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; width: 100%; }
[data-testid="stFileUploadDropzone"] { background: #0d0d14 !important; border: 2px dashed #1e1e2e !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🔬 Biocon Ltd. · AI Diagnostics Division</div>
    <div class="hero-title">Malaria Cell<br>Classifier</div>
    <div class="hero-sub">ResNet50 · Transfer Learning · Grad-CAM · Binary Classification</div>
</div>
""", unsafe_allow_html=True)

# ─── Model Loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "malaria_model.h5"
    GDRIVE_FILE_ID = "1OcxmOnvOruTvbMumaRZyvz1_RtcjrQLQ"
    if not os.path.exists(model_path):
        with st.spinner("⬇️ Downloading model from Google Drive (only once)..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                model_path,
                quiet=False,
                fuzzy=True
            )
    return tf.keras.models.load_model(model_path, compile=False)

# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
def make_gradcam(img_array, model):
    try:
        # Find last conv layer name
        last_conv_name = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_name = layer.name
                break
        if last_conv_name is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_cam(img_pil, heatmap):
    img_arr = np.array(img_pil.resize((128, 128)))
    h = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    colored = (jet[h] * 255).astype(np.uint8)
    blended = (colored * 0.4 + img_arr * 0.6).astype(np.uint8)
    return Image.fromarray(blended)

def preprocess(img_pil):
    img = img_pil.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ─── Layout ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    # Stats
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model Architecture</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-box"><div class="stat-val">ResNet50</div><div class="stat-label">Backbone</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-box"><div class="stat-val">128px</div><div class="stat-label">Input</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-box"><div class="stat-val">Binary</div><div class="stat-label">Task</div></div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Steps
    st.markdown('<div class="card-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step"><div class="step-num">1</div><div class="step-text">Upload a microscopy image of a red blood cell</div></div>
    <div class="step"><div class="step-num">2</div><div class="step-text">ResNet50 extracts deep visual features via transfer learning</div></div>
    <div class="step"><div class="step-num">3</div><div class="step-text">Model predicts <b style="color:#00ff88">Healthy</b> or <b style="color:#ff4466">Infected</b> with confidence score</div></div>
    <div class="step"><div class="step-num">4</div><div class="step-text">Grad-CAM heatmap shows which cell regions influenced the prediction</div></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload Cell Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a cell image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        img_pil = Image.open(uploaded_file).convert("RGB")
        st.image(img_pil, caption="Uploaded Image", use_container_width=True)
        run_btn = st.button("🔬 Run Diagnosis")
    else:
        st.markdown('<div style="text-align:center;padding:1rem 0;color:#6b6b8a;font-family:monospace;font-size:0.75rem;">Accepted: JPG · PNG · JPEG</div>', unsafe_allow_html=True)
        run_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">⚠️ RESEARCH USE ONLY — Not a certified medical device. Predictions must be validated by a qualified medical professional before clinical use.</div>', unsafe_allow_html=True)

# ─── Results ──────────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Diagnosis Results</div>', unsafe_allow_html=True)

    if uploaded_file and run_btn:
        with st.spinner("Loading model & running inference..."):
            model = load_model()
            img_array = preprocess(img_pil)
            prediction = float(model.predict(img_array, verbose=0)[0][0])

        is_infected = prediction > 0.5
        confidence = prediction if is_infected else (1 - prediction)
        conf_pct = f"{confidence * 100:.1f}%"

        if is_infected:
            st.markdown(f"""
            <div class="result-infected">
                <div style="font-size:2.5rem">🦟</div>
                <div class="result-label" style="color:#ff4466">INFECTED</div>
                <div class="result-conf">Parasite detected · Confidence: {conf_pct}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <div style="font-size:2.5rem">✅</div>
                <div class="result-label" style="color:#00ff88">HEALTHY</div>
                <div class="result-conf">No parasite detected · Confidence: {conf_pct}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">Confidence Breakdown</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Healthy Score", f"{(1-prediction)*100:.1f}%")
            st.progress(float(1 - prediction))
        with col2:
            st.metric("Infected Score", f"{prediction*100:.1f}%")
            st.progress(float(prediction))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">Grad-CAM Interpretability</div>', unsafe_allow_html=True)
        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap = make_gradcam(img_array, model)

        if heatmap is not None:
            cam_img = overlay_cam(img_pil, heatmap)
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_pil.resize((128, 128)), caption="Original", use_container_width=True)
            with c2:
                st.image(cam_img, caption="Grad-CAM Overlay", use_container_width=True)
            st.markdown('<div style="font-family:monospace;font-size:0.65rem;color:#6b6b8a;margin-top:0.5rem;">🔴 Red/Yellow = high attention &nbsp;·&nbsp; 🔵 Blue = low attention</div>', unsafe_allow_html=True)
        else:
            st.info("Grad-CAM not available for this model configuration.")

    elif uploaded_file and not run_btn:
        st.markdown('<div style="text-align:center;padding:4rem 2rem;color:#6b6b8a;"><div style="font-size:3rem">🔬</div><div style="font-family:monospace;font-size:0.75rem;letter-spacing:0.1em;margin-top:1rem;">IMAGE UPLOADED · PRESS RUN TO ANALYZE</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:4rem 2rem;color:#6b6b8a;"><div style="font-size:3rem">🧬</div><div style="font-family:monospace;font-size:0.75rem;letter-spacing:0.1em;margin-top:1rem;">AWAITING CELL IMAGE UPLOAD</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:2rem;margin-top:1rem;border-top:1px solid #1e1e2e;"><div style="font-family:monospace;font-size:0.6rem;color:#3a3a5a;letter-spacing:0.15em;">BIOCON LTD. · AI DIAGNOSTICS · MALARIA DETECTION · RESEARCH USE ONLY</div></div>', unsafe_allow_html=True)
