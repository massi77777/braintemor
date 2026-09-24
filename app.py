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
