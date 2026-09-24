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
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected",
}

ICONS = {
    "glioma": "🔴",
    "meningioma": "🟠",
    "pituitary": "🟡",
    "notumor": "🟢",
}

COLORS = {
    "glioma": "#e74c3c",
    "meningioma": "#e67e22",
    "pituitary": "#f1c40f",
    "notumor": "#2ecc71",
}

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 0.2rem;
    }

    /* Subtitle/caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        text-align: center;
        color: #b0bec5 !important;
    }

    /* Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 198, 255, 0.3);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }

    div[data-testid="stMetricLabel"] {
        color: #b0bec5 !important;
        font-size: 1rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #00c6ff !important;
        font-weight: 700 !important;
    }

    /* Image container */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(0, 198, 255, 0.2);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(0, 198, 255, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 198, 255, 0.8);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.95);
        border-right: 1px solid rgba(0, 198, 255, 0.2);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 198, 255, 0.4);
    }

    /* Info/warning boxes */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-left: 4px solid #00c6ff;
    }

    /* Bar chart container */
    [data-testid="stVegaLiteChart"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(0, 198, 255, 0.15);
    }

    /* Subheaders */
    h2, h3 {
        color: #e0f7fa !important;
    }

    /* Prediction badge */
    .prediction-badge {
        display: inline-block;
        padding: 0.6rem 1.4rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Divider */
    hr {
        border-color: rgba(0, 198, 255, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


# ---------- Header ----------
st.markdown("<h1>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#b0bec5; font-size:0.95rem;'>"
    "⚠️ Educational demo only — NOT a medical device. Must not be used for diagnosis."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

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

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🩺 About")
    st.markdown(
        """
        <div style="background: rgba(0,198,255,0.08); padding:1rem; border-radius:12px; 
                    border-left:3px solid #00c6ff; margin-bottom:1rem;">
        <p style="color:#e0f7fa; margin:0;">
        A <b>ConvNeXt-Small</b> model fine-tuned with <b>fastai</b> on the 
        <a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset" 
           style="color:#00c6ff;" target="_blank">Brain Tumor MRI Dataset</a>.
        It reached about <b>95% accuracy</b> on the dataset's test split.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Classes")
    for cls in learn.dls.vocab:
        icon = ICONS.get(str(cls), "•")
        st.markdown(f"{icon} `{cls}`")

    st.markdown("---")
    st.markdown(
        "<p style='color:#78909c; font-size:0.8rem;'>"
        "Built with Streamlit + fastai 🚀</p>",
        unsafe_allow_html=True,
    )

# ---------- Main content ----------
uploaded = st.file_uploader(
    "📤 Upload a brain MRI image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Drag and drop or click to browse your files.",
)

if uploaded:
    img = PILImage.create(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(img.to_thumb(400), use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        label, _, probs = learn.predict(img)

    label = str(label)
    confidence = probs.max().item()
    desc = DESCRIPTIONS.get(label, label)
    icon = ICONS.get(label, "•")
    color = COLORS.get(label, "#00c6ff")

    with col2:
        st.markdown("### 🎯 Prediction")
        st.markdown(
            f"""
            <div style="text-align:center; padding:1.5rem; background: rgba(255,255,255,0.04);
                        border-radius:16px; border:1px solid {color}55; margin-bottom:1rem;">
                <div class="prediction-badge" style="background:{color};">
                    {icon} {desc}
                </div>
                <div style="font-size:2.5rem; font-weight:800; color:{color}; margin-top:0.5rem;">
                    {confidence:.1%}
                </div>
                <div style="color:#b0bec5; font-size:0.9rem;">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence < 0.70:
            st.warning("⚠️ Low confidence. The image may be out of distribution or unclear.")

    st.markdown("---")
    st.markdown("### 📊 Class Probabilities")

    prob_dict = {f"{ICONS.get(str(c), '•')} {c}": float(p) for c, p in zip(learn.dls.vocab, probs)}
    st.bar_chart(prob_dict, use_container_width=True)

else:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem; background: rgba(255,255,255,0.03);
                    border-radius:20px; border:2px dashed rgba(0,198,255,0.3); margin-top:2rem;">
            <div style="font-size:4rem;">🧠</div>
            <h3 style="color:#e0f7fa;">Ready to Analyze</h3>
            <p style="color:#b0bec5;">
                Upload a brain MRI image (JPG or PNG) above to get an instant prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
MODEL_PATH = "model.pkl"

DESCRIPTIONS = {
    "glioma": "Glioma tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor detected",
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


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


st.title("🧠 Brain Tumor MRI Classifier")
st.caption(
    "Educational demo only. This is NOT a medical device and must not be used for diagnosis."
)

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
    st.header("About")
    st.write(
        "A ConvNeXt-Small model fine-tuned with fastai on the "
        "[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). "
        "It reached about 95% accuracy on the dataset's test split."
    )
    st.write("**Classes:** " + ", ".join(f"`{c}`" for c in learn.dls.vocab))

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = PILImage.create(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img.to_thumb(320), caption="Uploaded image")

    with st.spinner("Analyzing..."):
        label, _, probs = learn.predict(img)

    label = str(label)
    confidence = probs.max().item()

    with col2:
        st.subheader("Prediction")
        st.metric(DESCRIPTIONS.get(label, label), f"{confidence:.1%}")
        if confidence < 0.70:
            st.warning("Low confidence. The image may be out of distribution or unclear.")

    st.subheader("Class probabilities")
    st.bar_chart(dict(zip(map(str, learn.dls.vocab), probs.tolist())))
else:
    st.info("Upload a brain MRI image (JPG or PNG) to get a prediction.")
