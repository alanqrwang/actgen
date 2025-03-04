import torch
import torch.nn as nn

from . import layers

from collections.abc import Sequence
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from rssl.unet3d import UNet3D

h_dims = [32, 64, 64, 128, 128, 256, 256, 512]


class UNetEncoder(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super().__init__()
        self.dim = dim

        self.block1 = layers.ConvBlockDown(
            input_ch, h_dims[0], 1, norm_type, False, dim
        )
        self.block2 = layers.ConvBlockDown(
            h_dims[0], h_dims[1], 1, norm_type, True, dim
        )

        self.block3 = layers.ConvBlockDown(
            h_dims[1], h_dims[2], 1, norm_type, False, dim
        )
        self.block4 = layers.ConvBlockDown(
            h_dims[2], h_dims[3], 1, norm_type, True, dim
        )

        self.block5 = layers.ConvBlockDown(
            h_dims[3], h_dims[4], 1, norm_type, False, dim
        )
        self.block6 = layers.ConvBlockDown(
            h_dims[4], h_dims[5], 1, norm_type, True, dim
        )

        self.block7 = layers.ConvBlockDown(
            h_dims[5], h_dims[6], 1, norm_type, False, dim
        )
        # self.block8 = layers.ConvBlockDown(
        #     h_dims[6], h_dims[7], 1, norm_type, True, dim
        # )

        self.block9 = layers.ConvBlockDown(h_dims[6], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block9(out)
        return out


class UNetDecoder(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super().__init__()
        self.dim = dim

        self.block1 = layers.ConvBlockUp(input_ch, h_dims[6], 1, norm_type, False, dim)
        self.block2 = layers.ConvBlockUp(h_dims[6], h_dims[5], 1, norm_type, True, dim)

        self.block3 = layers.ConvBlockUp(h_dims[5], h_dims[4], 1, norm_type, False, dim)
        self.block4 = layers.ConvBlockUp(h_dims[4], h_dims[3], 1, norm_type, True, dim)

        self.block5 = layers.ConvBlockUp(h_dims[3], h_dims[2], 1, norm_type, False, dim)
        self.block6 = layers.ConvBlockUp(h_dims[2], h_dims[1], 1, norm_type, True, dim)

        self.block7 = layers.ConvBlockUp(h_dims[1], h_dims[0], 1, norm_type, False, dim)
        # self.block8 = layers.ConvBlockUp(h_dims[1], h_dims[0], 1, norm_type, True, dim)

        self.block9 = layers.ConvBlockUp(h_dims[0], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block9(out)
        return out


class UNetDecoderViT(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super().__init__()
        self.dim = dim

        self.block1 = layers.ConvBlockUp(input_ch, h_dims[6], 1, norm_type, False, dim)
        self.block2 = layers.ConvBlockUp(h_dims[6], h_dims[5], 1, norm_type, True, dim)

        self.block3 = layers.ConvBlockUp(h_dims[5], h_dims[4], 1, norm_type, False, dim)
        self.block4 = layers.ConvBlockUp(h_dims[4], h_dims[3], 1, norm_type, True, dim)

        self.block5 = layers.ConvBlockUp(h_dims[3], h_dims[2], 1, norm_type, False, dim)
        self.block6 = layers.ConvBlockUp(h_dims[2], h_dims[1], 1, norm_type, True, dim)

        self.block7 = layers.ConvBlockUp(h_dims[1], h_dims[0], 1, norm_type, False, dim)
        self.block8 = layers.ConvBlockUp(h_dims[0], h_dims[0], 1, norm_type, True, dim)

        self.block9 = layers.ConvBlockUp(h_dims[0], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        return out


class UNetDecoderHalf(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super().__init__()
        self.dim = dim

        self.block1 = layers.ConvBlockUp(input_ch, h_dims[6], 1, norm_type, False, dim)
        self.block2 = layers.ConvBlockUp(h_dims[6], h_dims[4], 1, norm_type, True, dim)

        self.block4 = layers.ConvBlockUp(h_dims[4], h_dims[2], 1, norm_type, True, dim)

        self.block6 = layers.ConvBlockUp(h_dims[2], h_dims[0], 1, norm_type, True, dim)

        self.block9 = layers.ConvBlockUp(h_dims[0], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block4(out)
        out = self.block6(out)
        out = self.block9(out)
        return out


class ConvNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        base_filters=16,
        norm_type="batch",
        latent_dims=(20, 24, 22),
    ):
        super(ConvNetDecoder, self).__init__()

        # Fully connected layer to reshape the feature vector into a 3D volume
        self.base_filters = base_filters
        self.latent_dims = latent_dims
        # self.fc = nn.Linear(input_dim, latent_dims[0] * latent_dims[1] * latent_dims[2])

        # self.init_conv = nn.Conv3d(1, base_filters * 16, kernel_size=1)

        # Define the upsampling (decoder) blocks with Conv3DTranspose
        self.up1 = layers.ConvBlockUp(
            in_channels,
            base_filters * 16,
            stride=1,
            norm_type=norm_type,
            upsample=True,
            dim=3,
        )
        self.up2 = layers.ConvBlockUp(
            base_filters * 16,
            base_filters * 8,
            stride=1,
            norm_type=norm_type,
            upsample=True,
            dim=3,
        )
        self.up3 = layers.ConvBlockUp(
            base_filters * 8,
            base_filters * 4,
            stride=1,
            norm_type=norm_type,
            upsample=True,
            dim=3,
        )
        self.up4 = layers.ConvBlockUp(
            base_filters * 4,
            base_filters * 2,
            stride=1,
            norm_type=norm_type,
            upsample=True,
            dim=3,
        )
        self.up5 = layers.ConvBlockUp(
            base_filters * 2,
            base_filters * 1,
            stride=1,
            norm_type=norm_type,
            upsample=True,
            dim=3,
        )

        # Final convolution layer to output desired number of channels (e.g., 1 for grayscale)
        self.final_conv = nn.Conv3d(base_filters * 1, output_channels, kernel_size=1)

    def forward(self, x):
        # Project the input feature vector to the initial 3D volume (e.g., 16x16x16)
        # x = self.fc(z)
        # x = x.view(
        # -1, 1, *self.latent_dims
        # )  # Reshape to [batch_size, channels, depth, height, width]
        # x = self.init_conv(x)

        # Upsample from 16x16x16 to 256x256x256
        x = self.up1(x)  # Now 40x...
        x = self.up2(x)  # Now 80x...
        x = self.up3(x)  # Now 160x...
        x = self.up4(x)  # Now 256x256x256
        x = self.up5(x)  # Now 256x256x256

        # Final convolution to produce the output volume
        output = self.final_conv(x)

        return output


class SingleLayerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        use_amp=False,
    ):
        super(SingleLayerDecoder, self).__init__()
        self.use_amp = use_amp
        self.conv = nn.ConvTranspose3d(
            input_dim, 3, kernel_size=(8, 8, 8), stride=(8, 8, 8)
        )

    def forward(self, x):
        with torch.amp.autocast(
            device_type="cuda", enabled=self.use_amp, dtype=torch.float16
        ):
            output = self.conv(x)

        return output


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


class MAEViTDecoder(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        out_channels: int,
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

        assert (
            patch_size[0] == patch_size[1] == patch_size[2]
        ), "Patch size must be equal in all dimensions"

        self.hidden_size = hidden_size
        self.classification = classification
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=1,  # not used
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.patch_embedding.patch_embeddings = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    mlp_dim,
                    num_heads,
                    dropout_rate,
                    qkv_bias,
                    save_attn,
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

        self.decoder_pred = nn.Linear(
            hidden_size, patch_size[0] ** spatial_dims * out_channels, bias=True
        )  # decoder to patch

    def forward(self, x):
        x = self.patch_embedding(x)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        # reshape to image
        x = x.view(x.shape[0], self.out_channels, *self.img_size)

        return x  # , hidden_states_out


class Unet3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        **kwargs,
    ):
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,  # Used by nnUNet
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            kwargs=kwargs,
        )

        self.encoder = model.encoder

    def forward(self, img):
        return self.encoder(img)
