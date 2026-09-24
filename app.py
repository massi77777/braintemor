import pathlib
import platform
import urllib.request

import streamlit as st
from fastai.vision.all import PILImage, load_learner

# =====================================================================
# ترقيعات التوافق (Shims)
# =====================================================================

# نموذج مُصدَّر على لينكس (Kaggle/Colab) يخزن كائنات PosixPath؛
# نجعله قابلاً للتحميل على ويندوز.
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# النموذج تم حفظه ببايثون 3.13+ حيث تعيش أصناف pathlib في pathlib._local
# الإصدارات الأقدم (3.11/3.12) لا تحتوي هذا الوحدة، لذا ننشئ alias.
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
MIN_MODEL_SIZE = 1_000_000  # أقل حجم معقول للنموذج (1 ميغابايت)

DESCRIPTIONS = {
    "glioma": "Glioma tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor detected",
}

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


# =====================================================================
# تحميل النموذج
# =====================================================================

def model_file_is_valid(path: pathlib.Path) -> bool:
    """التحقق من وجود الملف وحجمه الدنيا."""
    return path.exists() and path.stat().st_size >= MIN_MODEL_SIZE


def ensure_model_file() -> None:
    """
    استخدام model.pkl من المشروع (Git LFS)،
    أو تحميله من MODEL_URL المحدد في Streamlit secrets.
    """
    path = pathlib.Path(MODEL_PATH)
    if model_file_is_valid(path):
        return

    # محاولة التحميل من الرابط
    try:
        url = st.secrets["MODEL_URL"]
    except Exception:
        return  # لا يوجد رابط في secrets — سيُعالَج الخطأ لاحقاً في load_model

    try:
        with st.spinner("Downloading model (first run only)..."):
            urllib.request.urlretrieve(url, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download the model: `{e}`. Check MODEL_URL in the app secrets.")
        st.stop()

    # التحقق من نجاح التحميل فعلياً
    if not model_file_is_valid(pathlib.Path(MODEL_PATH)):
        st.error("The downloaded model file is missing or too small. Check MODEL_URL in the app secrets.")
        st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    ensure_model_file()
    try:
        learn = load_learner(MODEL_PATH, cpu=True)
    except UnboundLocalError:
        # fastai يخفي خطأ الاستيراد الحقيقي أثناء فك التسلسل —
        # نعيد التحميل لكشفه وعرضه.
        import pickle
        import torch
        torch.load(MODEL_PATH, map_location="cpu", pickle_module=pickle, weights_only=False)
        raise
    return learn


# =====================================================================
# واجهة المستخدم
# =====================================================================

st.title("🧠 Brain Tumor MRI Classifier")
st.caption(
    "Educational demo only. This is NOT a medical device and must not be used for diagnosis."
)

# تحميل النموذج مع معالجة الأخطاء
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
        f"`{MODEL_PATH}` was not found. Train the model with `train_and_export.py` "
        "and place the file next to `app.py` (or set MODEL_URL in the app secrets)."
    )
    st.stop()
except Exception as e:
    # أي خطأ آخر (ملف تالف، اختلاف إصدار fastai/torch...)
    st.error(f"Failed to load the model: `{type(e).__name__}: {e}`")
    st.stop()

# الشريط الجانبي
with st.sidebar:
    st.header("About")
    st.write(
        "A ConvNeXt-Small model fine-tuned with fastai on the "
        "[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). "
        "It reached about 95% accuracy on the dataset's test split."
    )
    st.write("**Classes:** " + ", ".join(f"`{c}`" for c in learn.dls.vocab))

# رفع الصورة
uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded:
    # --- التحقق من صلاحية الصورة ---
    try:
        img = PILImage.create(uploaded)
    except Exception as e:
        st.error(f"The uploaded file is not a valid image: `{e}`. Please upload a JPG or PNG MRI scan.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(img.to_thumb(320), caption="Uploaded image")

    # --- التوقع مع حماية من الأخطاء ---
    try:
        with st.spinner("Analyzing..."):
            label, _, probs = learn.predict(img)
    except Exception as e:
        st.error(f"Prediction failed: `{type(e).__name__}: {e}`")
        st.stop()

    label = str(label)
    confidence = float(probs.max().item())

    with col2:
        st.subheader("Prediction")
        st.metric(DESCRIPTIONS.get(label, label), f"{confidence:.1%}")
        if confidence < 0.70:
            st.warning("Low confidence. The image may be out of distribution or unclear.")

    st.subheader("Class probabilities")
    st.bar_chart(dict(zip(map(str, learn.dls.vocab), probs.tolist())))
else:
    st.info("Upload a brain MRI image (JPG or PNG) to get a prediction.")
