import torch
import torch.nn as nn


class AirlineSentimentModel(nn.Module):
    def __init__(self, structured_dim, tfidf_dim, num_classes):
        super().__init__()

        # Structured features branch
        self.structured_net = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )

        # Tweet text branch
        self.text_net = nn.Sequential(
            nn.Linear(tfidf_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Negative reason branch
        self.reason_net = nn.Sequential(
            nn.Linear(tfidf_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, numeric_and_ohe, text_data, reason_data):

        structured_out = self.structured_net(numeric_and_ohe)
        text_out = self.text_net(text_data)
        reason_out = self.reason_net(reason_data)

        fused = torch.cat([structured_out, text_out, reason_out], dim=1)

        return self.classifier(fused)