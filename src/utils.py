import torch
import random
import numpy as np


# -----------------------------
# Set seed for reproducibility
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Move batch to device (CPU/GPU)
# -----------------------------
def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# -----------------------------
# Simple training logger
# -----------------------------
def print_epoch_stats(epoch, loss):
    print(f"[Epoch {epoch}] Loss: {loss:.4f}")

# -----------------------------
# Save model checkpoint
# -----------------------------
def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# -----------------------------
# Load model checkpoint
# -----------------------------
def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")