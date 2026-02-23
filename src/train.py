import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import config

from data_loader import load_data
from preprocessing import setup_nltk, build_preprocessing_objects
from model import AirlineSentimentModel
from utils import set_seed, move_batch_to_device, print_epoch_stats, save_checkpoint


# -----------------------------
# 1. Reproducibility
# -----------------------------
set_seed(config.RANDOM_SEED)


# -----------------------------
# 2. Device (CPU / GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 3. Setup NLP resources
# -----------------------------
setup_nltk()


# -----------------------------
# 4. Load Data
# -----------------------------
dataset, train_loader, test_loader = load_data(config.DATA_PATH)


# -----------------------------
# 5. Build Preprocessing Objects
# -----------------------------
custom_collate, tfidf, sentiment_enc, airline_enc = build_preprocessing_objects(dataset)

train_loader.collate_fn = custom_collate
test_loader.collate_fn = custom_collate


# -----------------------------
# 6. Build Model
# -----------------------------
num_airlines = len(airline_enc.categories_[0])
num_classes = len(sentiment_enc.categories_[0])
structured_dim = 2 + num_airlines   # VERY IMPORTANT
tfidf_dim = config.TFIDF_MAX_FEATURES

model = AirlineSentimentModel(structured_dim, tfidf_dim, num_classes)
model = model.to(device)


# -----------------------------
# 7. Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# -----------------------------
# 8. Training Loop
# -----------------------------
train_losses = []
val_losses = []

for epoch in range(config.EPOCHS):
    # ------------------ TRAIN ------------------
    model.train()
    running_train_loss = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        numeric = batch["numeric_and_ohe"]
        text = batch["text_data"]
        reason = batch["reason_data"]

        outputs = model(
                    batch["numeric_and_ohe"],
                    batch["text_data"],
                    batch["reason_data"]
                )

        targets = torch.argmax(batch["targets"], dim=1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # ------------------ VALIDATION ------------------
    model.eval()
    running_val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)

            numeric = batch["numeric_and_ohe"]
            text = batch["text_data"]
            reason = batch["reason_data"]

            outputs = model(
                    batch["numeric_and_ohe"],
                    batch["text_data"],
                    batch["reason_data"]
                )

            targets = torch.argmax(batch["targets"], dim=1)

            loss = criterion(outputs, targets)
            running_val_loss += loss.item()

    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

plt.figure()
plt.plot(range(1, config.EPOCHS + 1), train_losses, label="Training Loss")
plt.plot(range(1, config.EPOCHS + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()
# -----------------------------
# 9. Save trained model
# -----------------------------
save_checkpoint(model, config.MODEL_SAVE_PATH)
print("Training Finished Successfully.")