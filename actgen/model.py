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

# ViT Encoder copied from rssl.net
class MAEViTEncoder(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(
                    nn.Linear(hidden_size, num_classes), nn.Tanh()
                )
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

        self.num_patches = self.patch_embedding.n_patches

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, img, mask_ratio):
        x = self.patch_embedding(img)

        x, token_mask, ids_restore = self.random_masking(x, mask_ratio)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, token_mask, ids_restore

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
        # print("tokens", tokens.shape)
        logits = self.transformer(tokens)
        return logits


# Example usage:
# model = VolumetricTrajectoryModel(in_channels=1, token_dim=128, num_timepoints=4, num_classes=10)
# input_scans = torch.randn(8, 4, 1, 64, 64, 64)  # Batch of 8, 4 timepoints, single channel, 64^3 volumes
# output = model(input_scans)

# Concatenate volumetric encoding from all timepoints and use an FC layer for classification
class VolumetricFCModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        token_dim=128,
        num_timepoints=4,
        num_classes=10,
    ):
        super(VolumetricFCModel, self).__init__()
        self.num_timepoints = num_timepoints # Expected number of timepoints
        self.encoder = VolumetricEncoder(in_channels, token_dim)
        # The FC layer takes the concatenated features from all timepoints:
        # output dimension = num_timepoints * token_dim
        self.fc = nn.Linear(num_timepoints * token_dim, num_classes)

    def forward(self, scans):
        # scans shape: [batch, available_timepoints, in_channels, D, H, W]
        batch_size, available_timepoints = scans.shape[:2]
        tokens = []
        for t in range(self.num_timepoints):
            if t < available_timepoints:
                # Process available scan
                token = self.encoder(scans[:, t])  # token shape: [batch, token_dim]
            else:
                # If missing, pad with zeros
                token = torch.zeros(batch_size, self.encoder.fc.out_features, device=scans.device)
            tokens.append(token)
        # Concatenate tokens along the feature dimension -> [batch, num_timepoints * token_dim]
        concat_features = torch.cat(tokens, dim=1)
        logits = self.fc(concat_features)
        return logits

# Model using ViT encoder with FC classification layer
class ViTFCModel(nn.Module):
    def __init__(
        self, 
        num_timepoints=4, 
        num_classes=2, 
        pretrained_path=None, 
        mask_ratio=0.0
    ):
        super(ViTFCModel, self).__init__()
        self.num_timepoints = num_timepoints
        
        # Instantiate the MAEViTEncoder with the same configuration as during pretraining.
        self.encoder = MAEViTEncoder(
            in_channels=1,
            img_size=[160, 192, 176],
            patch_size=[16, 16, 16],
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type="conv",
            pos_embed_type="sincos",
            classification=False,
            num_classes=0,
            dropout_rate=0.0,
            spatial_dims=3,
        )
        
        # Checkpoint loading
        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.encoder.load_state_dict(state_dict)
        
        # Remove classification head
        self.encoder.classification_head = nn.Identity()
        # FC layer concatenates features from all timepoints
        self.fc = nn.Linear(num_timepoints * self.encoder.hidden_size, num_classes)
        self.mask_ratio = mask_ratio

    def forward(self, scans):
        # scans shape: [batch, available_timepoints, in_channels, D, H, W]
        batch_size, available_timepoints = scans.shape[:2]
        tokens = []
        for t in range(self.num_timepoints):
            if t < available_timepoints:
                # Process available scan
                # The encoder expects input of shape [batch, in_channels, D, H, W] and a mask_ratio.
                # Encoder should returns (x, token_mask, ids_restore) since classification head is just identity matrix
                feat, _, _ = self.encoder(scans[:, t], self.mask_ratio)
            else:
                # If missing, pad with zeros
                token = torch.zeros(batch_size, self.encoder.fc.out_features, device=scans.device)
            tokens.append(token)
        # Concatenate tokens along the feature dimension -> [batch, num_timepoints * token_dim]
        concat_features = torch.cat(tokens, dim=1)
        logits = self.fc(concat_features)
        return logits