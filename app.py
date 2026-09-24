import pathlib
import platform
import urllib.request

import streamlit as st
from fastai.vision.all import PILImage, load_learner

# A model exported on Linux (Kaggle/Colab) stores PosixPath objects; make it loadable on Windows.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# The model was pickled on Python 3.13+, where pathlib classes live in `pathlib._local`.
# Older Python versions (like 3.11/3.12) don't have that module, so provide an alias.
try:
    import pathlib._local  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    _shim = types.ModuleType("pathlib._local")
    for _name in ("Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath", "PureWindowsPath"):
        setattr(_shim, _name, getattr(pathlib, _name))
    sys.modules["pathlib._local"] = _shim

MODEL_PATH = "model.pkl"

DESCRIPTIONS = {
    "glioma": "Glioma tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor detected",
}

st.set_page_config(page_title="Brain MRI • Tumor Classifier", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


def ensure_model_file():
    """Use model.pkl from the repo (Git LFS), or download it from MODEL_URL if set in Streamlit secrets."""
    path = pathlib.Path(MODEL_PATH)
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    try:
        url = st.secrets["MODEL_URL"]
    except Exception:
        return
    with st.spinner("Downloading model (first run only)..."):
        urllib.request.urlretrieve(url, MODEL_PATH)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    ensure_model_file()
    try:
        return load_learner(MODEL_PATH, cpu=True)
    except UnboundLocalError:
        # fastai hides the real ImportError raised while unpickling. Re-run the load to expose it.
        import pickle
        import torch
        torch.load(MODEL_PATH, map_location="cpu", pickle_module=pickle, weights_only=False)
        raise



# ---------- Modern UI ----------
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#07111f 0%,#0b1d32 52%,#06101d 100%);}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:3rem;}
.hero{padding:2.2rem 2.4rem;border:1px solid rgba(144,224,239,.18);border-radius:26px;
background:radial-gradient(circle at 90% 15%,rgba(0,180,216,.18),transparent 32%),
linear-gradient(135deg,rgba(3,31,55,.96),rgba(5,20,36,.96));box-shadow:0 18px 55px rgba(0,0,0,.28);margin-bottom:1.4rem;}
.hero-badge{display:inline-block;padding:.38rem .75rem;border-radius:999px;background:rgba(0,180,216,.12);
border:1px solid rgba(144,224,239,.25);color:#90e0ef;font-size:.82rem;font-weight:700;letter-spacing:.04em;margin-bottom:.8rem;}
.hero h1{margin:0;color:#f4fbff;font-size:clamp(2rem,4vw,3.2rem);line-height:1.08;letter-spacing:-.03em;}
.hero p{color:#b9d0dc;font-size:1.03rem;margin:.8rem 0 0;max-width:760px;line-height:1.65;}
.section-title{color:#eaf8fc;font-size:1.15rem;font-weight:750;margin:1.2rem 0 .65rem;}
.upload-card,.result-card{border:1px solid rgba(144,224,239,.14);border-radius:20px;background:rgba(8,27,45,.72);
padding:1.15rem;box-shadow:0 10px 35px rgba(0,0,0,.18);}
.result-label{color:#9ec2d0;font-size:.86rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.25rem;}
.result-value{color:#90e0ef;font-size:1.65rem;font-weight:800;line-height:1.2;}
.confidence{color:#f1f8fb;font-size:2.2rem;font-weight:850;margin-top:.35rem;}
.disclaimer{margin-top:1.2rem;padding:.9rem 1rem;border-left:4px solid #00b4d8;border-radius:10px;
background:rgba(0,180,216,.07);color:#b9d0dc;font-size:.88rem;line-height:1.55;}
[data-testid="stFileUploader"]{border:1px dashed rgba(144,224,239,.38);border-radius:16px;padding:.35rem;background:rgba(3,15,28,.38);}
[data-testid="stSidebar"]{background:#061525;border-right:1px solid rgba(144,224,239,.10);}
.sidebar-title{color:#90e0ef;font-size:1.15rem;font-weight:800;}
.sidebar-text{color:#a9c2ce;line-height:1.6;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-badge">AI • BRAIN MRI ANALYSIS</div>
  <h1>🧠 Brain Tumor MRI Classifier</h1>
  <p>Upload a brain MRI image and let the trained AI model classify it into one of four categories.</p>
</div>
<div class="disclaimer"><strong>⚠️ Educational demonstration only.</strong>
This application is not a medical device and must not be used to diagnose, treat, or make clinical decisions.</div>
""", unsafe_allow_html=True)

try:
    learn = load_model()
except ModuleNotFoundError as e:
    st.error(f"Missing Python module: `{e.name}`. Add it to requirements.txt and reboot the app.")
    st.stop()
except FileNotFoundError:
    st.error(
        f"`{MODEL_PATH}` was not found. Train the model with `train_and_export.py` "
        "and place the file next to `app.py` (or set MODEL_URL in the app secrets)."
    )
    st.stop()

with st.sidebar:
    st.markdown('<div class="sidebar-title">About the model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">A ConvNeXt-Small model fine-tuned with fastai on the '
        '<a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset" target="_blank">Brain Tumor MRI Dataset</a>.<br><br>'
        'It reached about <strong>95% accuracy</strong> on the dataset test split.</div>',
        unsafe_allow_html=True
    )
    st.markdown("### 🧩 Classes")
    for cls in learn.dls.vocab:
        st.markdown(f"- `{cls}` — {DESCRIPTIONS.get(str(cls), str(cls))}")

st.markdown('<div class="section-title">📤 Upload an MRI image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image",
                            type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded:
    img = PILImage.create(uploaded)
    col1, col2 = st.columns([1.05, .95], gap="large")

    with col1:
        st.markdown('<div class="section-title">MRI image</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.image(img.to_thumb(520), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔎 Analyzing MRI image..."):
        label, _, probs = learn.predict(img)

    label = str(label)
    confidence = probs.max().item()

    with col2:
        st.markdown('<div class="section-title">AI prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">Detected category</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{DESCRIPTIONS.get(label, label)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label" style="margin-top:1rem;">Confidence</div>'
                    f'<div class="confidence">{confidence:.1%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if confidence < 0.70:
            st.warning("Low confidence. The image may be unclear or different from the images used to train the model.")

    st.markdown('<div class="section-title">📊 Class probabilities</div>', unsafe_allow_html=True)
    st.bar_chart(dict(zip(map(str, learn.dls.vocab), probs.tolist())), height=300, use_container_width=True)

else:
    st.markdown("""
    <div class="upload-card" style="text-align:center;padding:2.2rem;">
      <div style="font-size:2.5rem;">🩻</div>
      <div style="color:#eaf8fc;font-size:1.15rem;font-weight:750;margin-top:.5rem;">Your MRI image will appear here</div>
      <div style="color:#91acb8;margin-top:.35rem;">Upload a JPG, JPEG, or PNG file above to start the analysis.</div>
    </div>
    """, unsafe_allow_html=True)
