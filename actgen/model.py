import torch
import torch.nn as nn
import torch.nn.functional as F


# 3D CNN Encoder for a single scan
class VolumetricEncoder(nn.Module):
    def __init__(self, in_channels=1, token_dim=128):
        super(VolumetricEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # Global pooling to get [batch, 128, 1, 1, 1]
        )
        self.fc = nn.Linear(128, token_dim)

    def forward(self, x):
        # x shape: [batch, in_channels, D, H, W]
        x = self.encoder(x)  # [batch, 128, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        token = self.fc(x)  # [batch, token_dim]
        return token


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


# Full Model combining volumetric encoding per scan and temporal transformer
class VolumetricTrajectoryModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        token_dim=128,
        num_timepoints=4,
        num_classes=10,
        num_transformer_layers=4,
    ):
        super(VolumetricTrajectoryModel, self).__init__()
        self.encoder = VolumetricEncoder(in_channels, token_dim)
        self.transformer = TemporalTransformer(
            token_dim, num_timepoints, num_classes, num_layers=num_transformer_layers
        )

    def forward(self, scans):
        # scans shape: [batch, num_timepoints, in_channels, D, H, W]
        batch_size, num_timepoints = scans.shape[:2]
        tokens = []
        for t in range(num_timepoints):
            # Process each scan individually
            token = self.encoder(scans[:, t])
            tokens.append(token.unsqueeze(1))  # [batch, 1, token_dim]
        tokens = torch.cat(tokens, dim=1)  # [batch, num_timepoints, token_dim]
        print("tokens", tokens.shape)
        logits = self.transformer(tokens)
        return logits


# Example usage:
# model = VolumetricTrajectoryModel(in_channels=1, token_dim=128, num_timepoints=4, num_classes=10)
# input_scans = torch.randn(8, 4, 1, 64, 64, 64)  # Batch of 8, 4 timepoints, single channel, 64^3 volumes
# output = model(input_scans)
