import torch
import torch.nn as nn

class AirlineSentimentModelLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, structured_dim, num_classes, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()

        # -------- Structured Features Branch --------
        self.structured_net = nn.Sequential(
            nn.Linear(structured_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )

        # -------- Text Embedding + LSTM --------
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.text_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # bidirectional => hidden*2
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------- Reason Embedding + LSTM --------
        self.reason_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.reason_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.reason_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # -------- Fusion Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, numeric_and_ohe, text_seq, reason_seq):
        # ----- Structured features -----
        structured_out = self.structured_net(numeric_and_ohe)

        # ----- Text branch -----
        text_emb = self.text_embedding(text_seq)                  # [B, T, E]
        _, (h_text, _) = self.text_lstm(text_emb)                # h_text: [num_layers*2, B, H]
        text_feat = torch.cat([h_text[-2], h_text[-1]], dim=1)   # last layer, forward + backward
        text_out = self.text_fc(text_feat)

        # ----- Reason branch -----
        reason_emb = self.reason_embedding(reason_seq)
        _, (h_reason, _) = self.reason_lstm(reason_emb)
        reason_feat = torch.cat([h_reason[-2], h_reason[-1]], dim=1)
        reason_out = self.reason_fc(reason_feat)

        # ----- Concatenate all features -----
        fused = torch.cat([structured_out, text_out, reason_out], dim=1)

        return self.classifier(fused)