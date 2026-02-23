import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1️. Load and clean dataset
# -----------------------------
def load_dataset(csv_path):
    dataset = pd.read_csv(csv_path)

    # Drop redundant columns
    dataset = dataset.drop(dataset.columns[[0,6,7,8,9,11,12,13,14]], axis=1)

    # Fill missing values
    dataset = dataset.fillna(0)

    return dataset


# -----------------------------
# 2. Split into X and y
# -----------------------------
def split_features_labels(dataset):
    X = dataset.iloc[:, 1:5].values
    y = dataset.iloc[:, 0].values

    return train_test_split(X, y, test_size=0.2, random_state=0)


# -----------------------------
# 3. Torch Dataset Class
# -----------------------------
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# -----------------------------
# 4. Create DataLoaders
# -----------------------------
def create_dataloaders(X_train, X_test, y_train, y_test,
                       train_batch_size=1000,
                       test_batch_size=100):

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    return train_loader, test_loader
# -----------------------------
# 5. Master function (ENTRY POINT)
# -----------------------------
def load_data(csv_path):
    """
    This is the ONLY function other files should call.
    It handles everything internally.
    """

    dataset = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = split_features_labels(dataset)

    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test
    )

    return dataset, train_loader, test_loader