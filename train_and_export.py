"""Run this on Kaggle or Colab with a GPU. It trains the model and saves model.pkl."""
# !pip install -q fastai kagglehub timm
import kagglehub
from fastai.vision.all import *

root = first(
    Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")).rglob("Training")
).parent

dls = ImageDataLoaders.from_folder(
    root, train="Training", valid="Testing",
    item_tfms=Resize(256), batch_tfms=aug_transforms(size=224), bs=32,
)


class ExportEachEpoch(Callback):
    """Saves the model after every epoch so you can stop training at any time."""
    def after_epoch(self):
        self.learn.export("model.pkl")


learn = vision_learner(dls, "convnext_small", metrics=accuracy).to_fp16()
learn.fine_tune(5, cbs=ExportEachEpoch())

preds, y = learn.tta()
print(f"Accuracy: {accuracy(preds, y).item():.2%}")

learn.export("model.pkl")
print("Saved model.pkl")
