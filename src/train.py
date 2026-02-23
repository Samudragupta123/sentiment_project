import torch
import torch.nn as nn
import torch.optim as optim

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
tfidf_dim = config.TFIDF_MAX_FEATURES

model = AirlineSentimentModel(num_airlines, tfidf_dim, num_classes)
model = model.to(device)


# -----------------------------
# 7. Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# -----------------------------
# 8. Training Loop
# -----------------------------
for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(batch)

        # convert one-hot → class index
        targets = torch.argmax(batch["targets"], dim=1)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print_epoch_stats(epoch + 1, total_loss)


print("Training Finished.")
# -----------------------------
# 9. Save trained model
# -----------------------------
save_checkpoint(model, config.MODEL_SAVE_PATH)