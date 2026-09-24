# Brain Tumor MRI Classifier

Streamlit app that classifies brain MRI images into `glioma`, `meningioma`, `pituitary` or `notumor`.
**Educational use only. Not for medical diagnosis.**

## Files

```
app.py               Streamlit app
train_and_export.py  Training script (run on Kaggle/Colab with a GPU)
requirements.txt     Dependencies (CPU-only PyTorch to keep the install small)
```

## 1. Train and export the model

Run `train_and_export.py` on Kaggle or Colab with a GPU enabled, then download `model.pkl`.

## 2. Run locally (optional)

Put `model.pkl` next to `app.py`, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud

1. Create a **public** GitHub repository and upload `app.py`, `requirements.txt`, `README.md`
   and `train_and_export.py` (Add file > Upload files).
2. Create a GitHub **Release** (tag `v1`) and attach `model.pkl` to it.
   Copy the download link of `model.pkl`.
3. Go to https://share.streamlit.io, click **Create app**, and select the repo, branch `main`
   and main file `app.py`.
4. In **Advanced settings**, choose Python 3.11 or 3.12 and paste this into **Secrets**:
   ```toml
   MODEL_URL = "https://github.com/<user>/<repo>/releases/download/v1/model.pkl"
   ```
5. Click **Deploy**. The model is downloaded on the first run only.

## Troubleshooting

- **Out of memory** (free apps have about 1 GB of RAM): retrain with `convnext_tiny` or `resnet34`.
- Keep fastai/torch versions close to those used for training. If loading fails, pin the
  versions from your training environment in `requirements.txt`.
- `PosixPath` errors on Windows are already handled in `app.py`.
