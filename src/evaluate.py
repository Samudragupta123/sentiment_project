import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

import config

from data_loader import load_data
from preprocessing import setup_nltk, build_preprocessing_objects
from model import AirlineSentimentModel
from utils import move_batch_to_device, load_checkpoint


# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 2. Setup NLP
# -----------------------------
setup_nltk()


# -----------------------------
# 3. Load Data
# -----------------------------
dataset, train_loader, test_loader = load_data(config.DATA_PATH)


# -----------------------------
# 4. Rebuild preprocessing (MUST match training)
# -----------------------------
custom_collate, tfidf, sentiment_enc, airline_enc = build_preprocessing_objects(dataset)

test_loader.collate_fn = custom_collate


# -----------------------------
# 5. Recreate model architecture
# -----------------------------
num_airlines = len(airline_enc.categories_[0])
num_classes = len(sentiment_enc.categories_[0])
tfidf_dim = config.TFIDF_MAX_FEATURES

model = AirlineSentimentModel(num_airlines, tfidf_dim, num_classes)


# -----------------------------
# 6. Load trained weights
# -----------------------------
load_checkpoint(model, config.MODEL_SAVE_PATH, device)


# -----------------------------
# 7. Metrics
# -----------------------------
accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)


# -----------------------------
# 8. Evaluation Loop
# -----------------------------
model.eval()

with torch.no_grad():
    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        targets = torch.argmax(batch["targets"], dim=1)

        accuracy.update(preds, targets)
        precision.update(preds, targets)
        recall.update(preds, targets)
        f1.update(preds, targets)
        confmat.update(preds, targets)


# -----------------------------
# 9. Print Results
# -----------------------------
print("\nEvaluation Results")
print("------------------")
print("Accuracy :", accuracy.compute().item())
print("Precision:", precision.compute().item())
print("Recall   :", recall.compute().item())
print("F1 Score :", f1.compute().item())
print("\nConfusion Matrix:\n", confmat.compute())