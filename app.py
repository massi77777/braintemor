import pathlib
import platform
import urllib.request
import tempfile
import os

import streamlit as st
from fastai.vision.all import PILImage, load_learner


# =========================================================
# Fix pathlib compatibility between Windows / Linux
# =========================================================

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


# Python 3.11/3.12 compatibility with models exported
# from newer Python versions
try:
    import pathlib._local
except ModuleNotFoundError:
    import sys
    import types

    _shim = types.ModuleType("pathlib._local")

    for _name in (
        "Path",
        "PosixPath",
        "WindowsPath",
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
    ):
        setattr(_shim, _name, getattr(pathlib, _name))

    sys.modules["pathlib._local"] = _shim


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "model.pkl"

DESCRIPTIONS = {
    "glioma": "Glioma tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor detected",
}


# =========================================================
# Streamlit page configuration
# =========================================================

st.set_page_config(
    page_title="Brain MRI • Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Modern UI
# =========================================================

st.markdown(
    """
<style>

/* =========================
   Main background
   ========================= */

.stApp {
    background:
        radial-gradient(
            circle at 90% 10%,
            rgba(0, 180, 216, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07111f 0%,
            #0b1d32 52%,
            #06101d 100%
        );
}


/* =========================
   Main container
   ========================= */

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}


/* =========================
   Hero section
   ========================= */

.hero {
    padding: 2.3rem 2.5rem;

    border: 1px solid rgba(144, 224, 239, 0.18);

    border-radius: 26px;

    background:
        radial-gradient(
            circle at 90% 15%,
            rgba(0, 180, 216, 0.18),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            rgba(3, 31, 55, 0.96),
            rgba(5, 20, 36, 0.96)
        );

    box-shadow:
        0 18px 55px rgba(0, 0, 0, 0.28);

    margin-bottom: 1.4rem;
}


.hero-badge {
    display: inline-block;

    padding: 0.38rem 0.75rem;

    border-radius: 999px;

    background: rgba(0, 180, 216, 0.12);

    border: 1px solid rgba(144, 224, 239, 0.25);

    color: #90e0ef;

    font-size: 0.82rem;

    font-weight: 700;

    letter-spacing: 0.04em;

    margin-bottom: 0.8rem;
}


.hero h1 {
    margin: 0;

    color: #f4fbff;

    font-size: clamp(2rem, 4vw, 3.2rem);

    line-height: 1.08;

    letter-spacing: -0.03em;
}


.hero p {
    color: #b9d0dc;

    font-size: 1.03rem;

    margin: 0.8rem 0 0;

    max-width: 760px;

    line-height: 1.65;
}


/* =========================
   Section titles
   ========================= */

.section-title {
    color: #eaf8fc;

    font-size: 1.15rem;

    font-weight: 750;

    margin: 1.2rem 0 0.65rem;
}


/* =========================
   Cards
   ========================= */

.upload-card,
.result-card {

    border: 1px solid rgba(144, 224, 239, 0.14);

    border-radius: 20px;

    background: rgba(8, 27, 45, 0.72);

    padding: 1.15rem;

    box-shadow:
        0 10px 35px rgba(0, 0, 0, 0.18);
}


/* =========================
   Result
   ========================= */

.result-label {

    color: #9ec2d0;

    font-size: 0.86rem;

    text-transform: uppercase;

    letter-spacing: 0.08em;

    margin-bottom: 0.25rem;
}


.result-value {

    color: #90e0ef;

    font-size: 1.65rem;

    font-weight: 800;

    line-height: 1.2;
}


.confidence {

    color: #f1f8fb;

    font-size: 2.2rem;

    font-weight: 850;

    margin-top: 0.35rem;
}


/* =========================
   Disclaimer
   ========================= */

.disclaimer {

    margin-top: 1.2rem;

    padding: 0.9rem 1rem;

    border-left: 4px solid #00b4d8;

    border-radius: 10px;

    background: rgba(0, 180, 216, 0.07);

    color: #b9d0dc;

    font-size: 0.88rem;

    line-height: 1.55;
}


/* =========================
   File uploader
   ========================= */

[data-testid="stFileUploader"] {

    border: 1px dashed rgba(144, 224, 239, 0.38);

    border-radius: 16px;

    padding: 0.35rem;

    background: rgba(3, 15, 28, 0.38);
}


/* =========================
   Sidebar
   ========================= */

[data-testid="stSidebar"] {

    background: #061525;

    border-right: 1px solid rgba(144, 224, 239, 0.10);
}


.sidebar-title {

    color: #90e0ef;

    font-size: 1.15rem;

    font-weight: 800;
}


.sidebar-text {

    color: #a9c2ce;

    line-height: 1.6;

    font-size: 0.9rem;
}


/* =========================
   Metrics
   ========================= */

[data-testid="stMetric"] {

    background: rgba(3, 15, 28, 0.35);

    border: 1px solid rgba(144, 224, 239, 0.10);

    border-radius: 16px;

    padding: 0.8rem;
}


/* =========================
   Mobile
   ========================= */

@media (max-width: 768px) {

    .hero {
        padding: 1.5rem;
    }

    .hero h1 {
        font-size: 2rem;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Hero
# =========================================================

st.markdown(
    """
<div class="hero">

    <div class="hero-badge">
        AI • BRAIN MRI ANALYSIS
    </div>

    <h1>
        🧠 Brain Tumor MRI Classifier
    </h1>

    <p>
        Upload a brain MRI image and let the trained AI model
        classify it into one of four categories.
    </p>

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Medical disclaimer
# =========================================================

st.markdown(
    """
<div class="disclaimer">

<strong>⚠️ Educational demonstration only.</strong>

This application is not a medical device and must not be used
to diagnose, treat, or make clinical decisions.

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Model download
# =========================================================

def ensure_model_file():

    """
    Use model.pkl from the repository.

    If the model is not present locally and MODEL_URL
    exists in Streamlit secrets, download it.
    """

    path = pathlib.Path(MODEL_PATH)

    if path.exists() and path.stat().st_size > 1_000_000:
        return

    try:
        url = st.secrets["MODEL_URL"]

    except Exception:
        return

    with st.spinner("Downloading model for the first run..."):

        urllib.request.urlretrieve(
            url,
            MODEL_PATH
        )


# =========================================================
# Load model
# =========================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():

    ensure_model_file()

    try:

        return load_learner(
            MODEL_PATH,
            cpu=True
        )

    except UnboundLocalError:

        # fastai can sometimes hide the original ImportError
        # while loading a pickle file.

        import pickle
        import torch

        torch.load(
            MODEL_PATH,
            map_location="cpu",
            pickle_module=pickle,
            weights_only=False
        )

        raise


# =========================================================
# Load model safely
# =========================================================

try:

    learn = load_model()

except ModuleNotFoundError as e:

    st.error(
        f"Missing Python module: `{e.name}`. "
        "Add it to requirements.txt and reboot the app."
    )

    st.stop()

except FileNotFoundError:

    st.error(
        f"`{MODEL_PATH}` was not found. "
        "Train the model with `train_and_export.py` "
        "and place the file next to `app.py` "
        "(or set MODEL_URL in the app secrets)."
    )

    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">About the model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="sidebar-text">

A ConvNeXt-Small model fine-tuned with fastai
on the Brain Tumor MRI Dataset.

<br><br>

It reached about <strong>95% accuracy</strong>
on the dataset test split.

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("### 🧩 Classes")

    for cls in learn.dls.vocab:

        class_name = str(cls)

        description = DESCRIPTIONS.get(
            class_name,
            class_name
        )

        st.markdown(
            f"- `{class_name}` — {description}"
        )


# =========================================================
# Upload section
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload an MRI image</div>',
    unsafe_allow_html=True
)


uploaded = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# =========================================================
# Prediction
# =========================================================

if uploaded:

    # -----------------------------------------------------
    # Create PIL image for displaying it
    # -----------------------------------------------------

    img = PILImage.create(
        uploaded
    )


    col1, col2 = st.columns(
        [1.05, 0.95],
        gap="large"
    )


    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    with col1:

        st.markdown(
            '<div class="section-title">MRI image</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="upload-card">',
            unsafe_allow_html=True
        )

        st.image(
            img.to_thumb(520),
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # FIX:
    # Save uploaded image as a temporary file.
    #
    # FastAI receives the FILE PATH instead of directly
    # receiving the PILImage object.
    # -----------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as temp_file:

            temp_file.write(
                uploaded.getvalue()
            )

            temp_path = temp_file.name


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        with st.spinner(
            "🔎 Analyzing MRI image..."
        ):

            label, _, probs = learn.predict(
                temp_path
            )


    except Exception as e:

        st.error(
            "The model could not process this image."
        )

        st.exception(e)

        st.stop()


    finally:

        # -------------------------------------------------
        # Delete temporary image
        # -------------------------------------------------

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except Exception:
                pass


    # -----------------------------------------------------
    # Convert prediction
    # -----------------------------------------------------

    label = str(label)

    confidence = probs.max().item()


    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">AI prediction</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">'
            'Detected category'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="result-value">
    {DESCRIPTIONS.get(label, label)}
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            """
<div class="result-label"
     style="margin-top:1rem;">
    Confidence
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="confidence">
    {confidence:.1%}
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Low confidence warning
        # -------------------------------------------------

        if confidence < 0.70:

            st.warning(
                "Low confidence. The image may be unclear "
                "or different from the images used to train "
                "the model."
            )


    # =====================================================
    # Probabilities
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Class probabilities'
        '</div>',
        unsafe_allow_html=True
    )


    probabilities = dict(
        zip(
            map(
                str,
                learn.dls.vocab
            ),
            probs.tolist()
        )
    )


    st.bar_chart(
        probabilities,
        height=300,
        use_container_width=True
    )


# =========================================================
# Empty state
# =========================================================

else:

    st.markdown(
        """
<div class="upload-card"
     style="text-align:center;padding:2.2rem;">

    <div style="font-size:2.5rem;">
        🩻
    </div>

    <div style="
        color:#eaf8fc;
        font-size:1.15rem;
        font-weight:750;
        margin-top:.5rem;
    ">
        Your MRI image will appear here
    </div>

    <div style="
        color:#91acb8;
        margin-top:.35rem;
    ">
        Upload a JPG, JPEG, or PNG file above
        to start the analysis.
    </div>

</div>
""",
        unsafe_allow_html=True
    )
