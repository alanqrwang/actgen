import torch
import torch.nn as nn
import torch.nn.functional as F


# Transformer-based model for temporal dynamics
class TemporalTransformer(nn.Module):
    def __init__(
        self, token_dim=128, num_timepoints=4, num_classes=10, num_layers=4, nhead=8
    ):
        super(TemporalTransformer, self).__init__()
        self.token_dim = token_dim

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        # Positional encoding for timepoints (including CLS)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_timepoints + 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(token_dim, num_classes)

    def forward(self, tokens):
        # tokens shape: [batch, num_timepoints, token_dim]
        batch_size = tokens.size(0)

        # Prepare CLS token and append to tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, token_dim]
        tokens = torch.cat(
            (cls_tokens, tokens), dim=1
        )  # [batch, num_timepoints+1, token_dim]

        # Add positional embeddings
        tokens = tokens + self.pos_embedding  # [batch, num_timepoints+1, token_dim]

        # Transformer expects shape [sequence_length, batch, token_dim]
        tokens = tokens.transpose(0, 1)
        out = self.transformer(tokens)  # [seq_len, batch, token_dim]

        # Use the CLS token (first token) for classification
        cls_out = out[0]
        logits = self.fc(cls_out)
        return logits


# Transformer-based model for temporal dynamics
class TemporalFC(nn.Module):
    def __init__(self, token_dim=128, num_timepoints=4, num_classes=10):
        super(TemporalFC, self).__init__()

        self.fc = nn.Linear(num_timepoints * token_dim, num_classes)
        self.token_dim = token_dim

    def forward(self, tokens):
        # tokens shape: [batch, num_timepoints, token_dim]
        batch_size, num_timepoints, token_dim = tokens.shape
        flat_features = tokens.view(batch_size, num_timepoints * token_dim)

        logits = self.fc(flat_features)
        return logits


# Full Model combining volumetric encoding per scan and temporal classifier
class VolumetricTemporalModel(nn.Module):
    def __init__(self, encoder, temporal_classifier, token_dim=128):
        super(VolumetricTemporalModel, self).__init__()
        self.encoder = encoder
        self.temporal_classifier = temporal_classifier
        self.missing_token = nn.Parameter(torch.zeros(1, token_dim))

    def forward(self, scans):
        # scans shape: [batch, num_timepoints, in_channels, D, H, W]
        batch_size, num_timepoints = scans.shape[:2]
        tokens = []
        for t in range(num_timepoints):
            scan = scans[:, t]
            # Process each scan individually
            if torch.isnan(scan).all():
                token = self.missing_token.expand(batch_size, -1)
            else:
                token = self.encoder(scan)
            tokens.append(token.unsqueeze(1))  # [batch, 1, token_dim]
        tokens = torch.cat(tokens, dim=1)  # [batch, num_timepoints, token_dim]
        # print("tokens", tokens.shape)
        logits = self.temporal_classifier(tokens)
        return logits
