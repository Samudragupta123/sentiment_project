# -----------------------------
# DATA PATHS
# -----------------------------
DATA_PATH = "data/raw_dataset.csv"


# -----------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------
BATCH_SIZE_TRAIN = 1000
BATCH_SIZE_TEST = 100
EPOCHS = 10
LEARNING_RATE = 1e-3


# -----------------------------
# NLP SETTINGS
# -----------------------------
TFIDF_MAX_FEATURES = 5000


# -----------------------------
# MODEL SETTINGS
# -----------------------------
META_HIDDEN_DIM = 64
TEXT_HIDDEN_DIM = 128
REASON_HIDDEN_DIM = 128
FUSION_DIM = 128


# -----------------------------
# RANDOMNESS CONTROL
# -----------------------------
RANDOM_SEED = 42


# -----------------------------
# CHECKPOINT SETTINGS
# -----------------------------
MODEL_SAVE_PATH = "checkpoints/saved_model.pth"