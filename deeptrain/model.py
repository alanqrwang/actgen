import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConcatTokens
from monai.networks.blocks.crossattention import CrossAttentionBlock

from rssl.utils import align_img, displacement2pytorchflow
from rssl import loss_ops


class VoxelMorph(nn.Module):
    def __init__(
        self,
        backbone,
        reg_final_conv,
        dim=3,
        use_amp=False,
        use_checkpoint=False,
        max_train_seg_channels=None,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super(RSSL, self).__init__()
        self.backbone = backbone
        self.dim = dim
        self.use_amp = use_amp
        self.use_checkpoint = use_checkpoint
        self.max_train_seg_channels = max_train_seg_channels
        self.reg_final_conv = reg_final_conv

    def forward(self, img_f, img_m, img_template):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img_m.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()
        imgs = torch.cat([img_m, img_f], dim=1)

        with torch.amp.autocast(
            device_type="cuda", enabled=self.use_amp, dtype=torch.float16
        ):
            output = self.backbone(imgs)

            output = self.reg_final_conv(output)

        result_dict = {
            "disp": output,
        }

        # Dictionary of results
        align_time = time.time() - start_time
        result_dict["time"] = (align_time,)
        return result_dict


class FinetuneModelCNN(nn.Module):
    def __init__(
        self, encoder, output_dim, dim=3, use_checkpoint=False, pred_head_type=None
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if pred_head_type and pred_head_type == "linear":
            self.pred_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim),
            )
        elif pred_head_type and pred_head_type == "mlp":
            self.pred_head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
        else:
            self.pred_head = None

    def forward(self, img):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img.shape[1] == 1, "Image dimension must be 1"

        result_dict = {}

        enc_output = self.encoder(img)
        if self.pred_head:
            pred_output = self.pred_head(enc_output)
        result_dict = {"pred_out": pred_output}
        return result_dict


class FinetuneModelViT(nn.Module):
    def __init__(
        self, encoder, output_dim, dim=3, use_checkpoint=False, pred_head_type=None
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.pred_head_type = pred_head_type
        if pred_head_type and pred_head_type == "gap":
            self.pred_head = nn.Sequential(
                nn.Linear(768, output_dim),
            )
        else:
            self.pred_head = None

    def forward(self, img):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img.shape[1] == 1, "Image dimension must be 1"

        result_dict = {}

        enc_output, _ = self.encoder(img)
        if self.pred_head_type == "gap":
            pred_output = self.pred_head(enc_output.mean(1))
        else:
            pred_output = enc_output
        result_dict = {"pred_out": pred_output}
        return result_dict


class SSL(nn.Module):
    def __init__(
        self,
        encoder,
        reg_decoder=None,
        recon_decoder=None,
        seg_decoder=None,
        use_checkpoint=False,
        num_proj_dim=0,
        num_proj_channels=0,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.num_proj_dim = num_proj_dim
        self.num_proj_channels = num_proj_channels
        self.use_checkpoint = use_checkpoint
        if num_proj_channels > 0:
            self.conv_projection = nn.Conv3d(256, num_proj_channels, kernel_size=1)
        if num_proj_dim > 0:
            self.linear_projection = nn.Linear(256, num_proj_dim)
        self.reg_decoder = reg_decoder
        self.recon_decoder = recon_decoder
        self.seg_decoder = seg_decoder

    def forward(self, img):
        """Forward pass for one mini-batch step.

        :param img: image
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()

        # Encoder
        if self.use_checkpoint:
            enc_out = torch.utils.checkpoint.checkpoint(
                self.encoder, img, use_reentrant=False
            )
        else:
            enc_out = self.encoder(img)

        # Handle single or multiple outputs from the encoder
        if isinstance(enc_out, tuple):
            enc_out, aux_out = enc_out
        else:
            enc_out = enc_out
            aux_out = None

        avgpool_out = (
            F.adaptive_avg_pool3d(enc_out, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        )

        # Latent projections
        if self.num_proj_channels > 0:
            proj_out = self.conv_projection(enc_out)
        else:
            proj_out = enc_out

        if self.num_proj_dim > 0:
            proj_lin = self.linear_projection(avgpool_out)
        else:
            proj_lin = None

        # Decoders
        dec_in = proj_out if aux_out is None else (proj_out, aux_out)
        if self.seg_decoder is not None:
            if self.use_checkpoint:
                seg_out = torch.utils.checkpoint.checkpoint(
                    self.seg_decoder, dec_in, use_reentrant=False
                )
            else:
                seg_out = self.seg_decoder(dec_in)
        else:
            seg_out = None

        if self.recon_decoder is not None:
            if self.use_checkpoint:
                recon_out = torch.utils.checkpoint.checkpoint(
                    self.recon_decoder,
                    dec_in,
                    use_reentrant=False,
                )
            else:
                recon_out = self.recon_decoder(dec_in)
        else:
            recon_out = None

        result_dict = {
            "enc_out": enc_out,
            "avgpool_out": avgpool_out,
            "proj_out": proj_out,
            "proj_lin": proj_lin,
            "seg_out": seg_out,
            "recon_out": recon_out,
        }

        # Dictionary of results
        align_time = time.time() - start_time
        result_dict["time"] = (align_time,)
        return result_dict


class RSSL(nn.Module):
    def __init__(
        self,
        encoder,
        reg_decoder=None,
        recon_decoder=None,
        seg_decoder=None,
        use_checkpoint=False,
        num_proj_dim=0,
        num_proj_channels=0,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.num_proj_dim = num_proj_dim
        self.num_proj_channels = num_proj_channels
        self.use_checkpoint = use_checkpoint
        if num_proj_channels > 0:
            self.conv_projection = nn.Conv3d(256, num_proj_channels, kernel_size=1)
        if num_proj_dim > 0:
            self.linear_projection = nn.Linear(256, num_proj_dim)
        self.reg_decoder = reg_decoder
        self.recon_decoder = recon_decoder
        self.seg_decoder = seg_decoder

    def forward(self, img_m, img_f):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img_m.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()

        # Encoder
        if self.use_checkpoint:
            enc_out_m = torch.utils.checkpoint.checkpoint(
                self.encoder, img_m, use_reentrant=False
            )
            enc_out_f = torch.utils.checkpoint.checkpoint(
                self.encoder, img_f, use_reentrant=False
            )
        else:
            enc_out_m = self.encoder(img_m)
            enc_out_f = self.encoder(img_f)

        avgpool_out_m = (
            F.adaptive_avg_pool3d(enc_out_m, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        )
        avgpool_out_f = (
            F.adaptive_avg_pool3d(enc_out_f, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        )

        # Latent projections
        if self.num_proj_channels > 0:
            proj_out_m = self.conv_projection(enc_out_m)
            proj_out_f = self.conv_projection(enc_out_f)
        else:
            proj_out_m = enc_out_m
            proj_out_f = enc_out_f

        if self.num_proj_dim > 0:
            proj_lin_m = self.linear_projection(avgpool_out_m)
            proj_lin_f = self.linear_projection(avgpool_out_f)
        else:
            proj_lin_m = None
            proj_lin_f = None

        # Decoders
        if self.seg_decoder is not None:
            if self.use_checkpoint:
                seg_out_m = torch.utils.checkpoint.checkpoint(
                    self.seg_decoder, proj_out_m, use_reentrant=False
                )
                seg_out_f = torch.utils.checkpoint.checkpoint(
                    self.seg_decoder, proj_out_f, use_reentrant=False
                )
            else:
                seg_out_m = self.seg_decoder(proj_out_m)
                seg_out_f = self.seg_decoder(proj_out_f)
        else:
            seg_out_m = None
            seg_out_f = None

        if self.reg_decoder is not None:
            if self.use_checkpoint:
                reg_out = torch.utils.checkpoint.checkpoint(
                    self.reg_decoder,
                    torch.cat([proj_out_m, proj_out_f], dim=1),
                    use_reentrant=False,
                )
            else:
                reg_out = self.reg_decoder(torch.cat([proj_out_m, proj_out_f], dim=1))
        else:
            reg_out = None
        if self.recon_decoder is not None:
            if self.use_checkpoint:
                recon_out_m = torch.utils.checkpoint.checkpoint(
                    self.recon_decoder,
                    proj_out_m,
                    use_reentrant=False,
                )
                recon_out_f = torch.utils.checkpoint.checkpoint(
                    self.recon_decoder,
                    proj_out_f,
                    use_reentrant=False,
                )
            else:
                recon_out_m = self.recon_decoder(proj_out_m)
                recon_out_f = self.recon_decoder(proj_out_f)
        else:
            recon_out_m = None
            recon_out_f = None

        result_dict = {
            "enc_out_m": enc_out_m,
            "enc_out_f": enc_out_f,
            "avgpool_out_m": avgpool_out_m,
            "avgpool_out_f": avgpool_out_f,
            "proj_out_m": proj_out_m,
            "proj_out_f": proj_out_f,
            "proj_lin_m": proj_lin_m,
            "proj_lin_f": proj_lin_f,
            "seg_out_m": seg_out_m,
            "seg_out_f": seg_out_f,
            "reg_out": reg_out,
            "recon_out_m": recon_out_m,
            "recon_out_f": recon_out_f,
        }

        # Dictionary of results
        align_time = time.time() - start_time
        result_dict["time"] = (align_time,)
        return result_dict


class SSLMAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        use_checkpoint=False,
        mask_ratio=0.75,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__()
        self.encoder = encoder
        self.use_checkpoint = use_checkpoint
        self.decoder = decoder
        encoder_hidden_size = encoder.hidden_size
        decoder_hidden_size = decoder.hidden_size

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(
            encoder_hidden_size, decoder_hidden_size, bias=True
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.mask_ratio = mask_ratio

    def add_mask_tokens(self, x, ids_restore):
        """append mask tokens to sequence"""
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        return x_

    def forward_encoder(self, img):
        # Encoder
        if self.use_checkpoint:
            enc_out, pixel_mask, ids_restore = torch.utils.checkpoint.checkpoint(
                self.encoder, img, self.mask_ratio, use_reentrant=False
            )
        else:
            enc_out, pixel_mask, ids_restore = self.encoder(img, self.mask_ratio)
        return enc_out, pixel_mask, ids_restore

    def forward_decoder(self, x):
        # Decoder
        if self.use_checkpoint:
            recon_out = torch.utils.checkpoint.checkpoint(
                self.decoder,
                x,
                use_reentrant=False,
            )
        else:
            recon_out = self.decoder(x)
        return recon_out

    def forward(self, img):
        """Forward pass for one mini-batch step.

        :param img: image
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img.shape[1] == 1, "Image dimension must be 1"

        # Encoder
        enc_out, token_mask, ids_restore = self.forward_encoder(img)

        # embed tokens
        x = self.decoder_embed(enc_out)

        # append mask tokens to sequence
        x = self.add_mask_tokens(x, ids_restore)

        recon_out = self.forward_decoder(x)
        loss = self.forward_loss(img, recon_out)

        # Dictionary of results
        result_dict = {
            "loss": loss,
            "enc_out": enc_out,
            "recon_out": recon_out,
        }
        return result_dict

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        return F.mse_loss(pred, imgs)


class RSSLMAE(SSLMAE):
    def __init__(
        self,
        *args,
        mix_type="concat",
        grad_loss_weight=0.0,
        **kwargs,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__(*args, **kwargs)

        self.mix_type = mix_type
        self.grad_loss_weight = grad_loss_weight
        decoder_hidden_size = kwargs["decoder"].hidden_size
        if mix_type == "concat":
            self.token_mixer = ConcatTokens()
        elif mix_type == "crossattn":
            self.token_mixer = CrossAttentionBlock(
                hidden_size=decoder_hidden_size,
                num_heads=1,
            )
        else:
            raise ValueError(f"Invalid mix_type: {self.mix_type}")

        if mix_type == "concat":
            self.decoder_embed_2 = nn.Linear(
                decoder_hidden_size * 2, decoder_hidden_size, bias=True
            )
        else:
            self.decoder_embed_2 = nn.Linear(
                decoder_hidden_size, decoder_hidden_size, bias=True
            )

    def forward(self, img_m, img_f):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img_m.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()

        # Encoder
        if self.use_checkpoint:
            enc_out_m = torch.utils.checkpoint.checkpoint(
                self.encoder, img_m, self.mask_ratio, use_reentrant=False
            )
            enc_out_f = torch.utils.checkpoint.checkpoint(
                self.encoder, img_f, self.mask_ratio, use_reentrant=False
            )
        else:
            enc_out_m, _, ids_restore_m = self.forward_encoder(img_m)
            enc_out_f, _, ids_restore_f = self.forward_encoder(img_f)

        # Embed into decoder hidden size
        x_m = self.decoder_embed(enc_out_m)
        x_f = self.decoder_embed(enc_out_f)

        # Insert mask tokens into sequence
        x_m = self.add_mask_tokens(x_m, ids_restore_m)
        x_f = self.add_mask_tokens(x_f, ids_restore_f)

        # Mix tokens
        x = self.token_mixer(x_m, x_f)

        # Embed into decoder hidden size
        x = self.decoder_embed_2(x)

        # Decoder
        if self.use_checkpoint:
            reg_out = torch.utils.checkpoint.checkpoint(
                self.decoder,
                x,
                use_reentrant=False,
            )
        else:
            reg_out = self.decoder(x)

        loss, reg_loss, grad_loss, img_a = self.forward_loss(reg_out, img_m, img_f)

        result_dict = {
            "loss": loss,
            "reg_loss": reg_loss,
            "grad_loss": grad_loss,
            "enc_out_m": enc_out_m,
            "enc_out_f": enc_out_f,
            "recon_out": img_a,
            "grad_loss_weight": self.grad_loss_weight,
            "disp": reg_out,
        }

        # Dictionary of results
        align_time = time.time() - start_time
        result_dict["time"] = (align_time,)
        return result_dict

    def forward_loss(self, network_output, img_m, img_f):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        disp = network_output
        disp_permute = disp.permute(0, 2, 3, 4, 1)
        flow = displacement2pytorchflow(disp_permute, input_space="norm")
        flow = flow.float()
        img_a = align_img(flow, img_m)

        reg_loss = F.mse_loss(img_a, img_f)
        grad_loss = loss_ops.Grad(loss_mult=1.0)(None, disp)
        loss = reg_loss + self.grad_loss_weight * grad_loss
        return loss, reg_loss, grad_loss, img_a


class MultiRSSLMAE(SSLMAE):
    def __init__(
        self,
        *args,
        reg_decoder,
        mix_type="concat",
        grad_loss_weight=0.0,
        **kwargs,
    ):
        """KeyMorph pipeline in a single module. Used for training.

        :param backbone: Backbone network
        :param num_keypoints: Number of keypoints
        :param dim: Dimension
        :param keypoint_extractor: Keypoint extractor
        :param max_train_keypoints: Maximum number of keypoints to use during training
        """
        super().__init__(*args, **kwargs)

        self.reg_decoder = reg_decoder
        self.mix_type = mix_type
        self.grad_loss_weight = grad_loss_weight
        decoder_hidden_size = kwargs["decoder"].hidden_size
        if mix_type == "concat":
            self.token_mixer = ConcatTokens()
        elif mix_type == "crossattn":
            self.token_mixer = CrossAttentionBlock(
                hidden_size=decoder_hidden_size,
                num_heads=1,
            )
        else:
            raise ValueError(f"Invalid mix_type: {self.mix_type}")

        # Post-mixing embedding
        if mix_type == "concat":
            self.decoder_embed_2 = nn.Linear(
                decoder_hidden_size * 2, decoder_hidden_size, bias=True
            )
        else:
            self.decoder_embed_2 = nn.Linear(
                decoder_hidden_size, decoder_hidden_size, bias=True
            )

    def forward(self, img_m, img_f):
        """Forward pass for one mini-batch step.

        :param img_f, img_m: Fixed and moving images
        :param transform_type: str or tuple of str of keypoint alignment types. Used for finding registrations
            for multiple alignment types in one forward pass, without having to extract keypoints
            every time.

        :return res: Dictionary of results
        """
        assert img_m.shape[1] == 1, "Image dimension must be 1"

        start_time = time.time()

        # Encoder
        if self.use_checkpoint:
            enc_out_m = torch.utils.checkpoint.checkpoint(
                self.encoder, img_m, self.mask_ratio, use_reentrant=False
            )
            enc_out_f = torch.utils.checkpoint.checkpoint(
                self.encoder, img_f, self.mask_ratio, use_reentrant=False
            )
        else:
            enc_out_m, _, ids_restore_m = self.forward_encoder(img_m)
            enc_out_f, _, ids_restore_f = self.forward_encoder(img_f)

        # Embed into decoder hidden size
        x_m = self.decoder_embed(enc_out_m)
        x_f = self.decoder_embed(enc_out_f)

        # Insert mask tokens into sequence
        x_m = self.add_mask_tokens(x_m, ids_restore_m)
        x_f = self.add_mask_tokens(x_f, ids_restore_f)

        ######### Reconstruction #########
        # Recon decoder
        if self.use_checkpoint:
            recon_out_m = torch.utils.checkpoint.checkpoint(
                self.decoder,
                x_m,
                use_reentrant=False,
            )
            recon_out_f = torch.utils.checkpoint.checkpoint(
                self.decoder,
                x_f,
                use_reentrant=False,
            )
        else:
            recon_out_m = self.decoder(x_m)
            recon_out_f = self.decoder(x_f)

        loss_m = super().forward_loss(img_m, recon_out_m)
        loss_f = super().forward_loss(img_f, recon_out_f)
        recon_loss = loss_m + loss_f

        ######### Registration #########
        # Mix tokens
        x = self.token_mixer(x_m, x_f)

        # Embed into decoder hidden size
        x = self.decoder_embed_2(x)

        # Registration decoder
        if self.use_checkpoint:
            reg_out = torch.utils.checkpoint.checkpoint(
                self.reg_decoder,
                x,
                use_reentrant=False,
            )
        else:
            reg_out = self.reg_decoder(x)

        reg_loss, sim_loss, grad_loss, img_a = self.forward_loss(reg_out, img_m, img_f)
        loss = recon_loss + reg_loss

        result_dict = {
            "loss": loss,
            "enc_out_m": enc_out_m,
            "enc_out_f": enc_out_f,
            # Reconstruction
            "recon_loss": recon_loss,
            "recon_out_m": recon_out_m,
            "recon_out_f": recon_out_f,
            # Registraiton
            "reg_loss": reg_loss,
            "sim_loss": sim_loss,
            "grad_loss": grad_loss,
            "reg_out": img_a,
            "grad_loss_weight": self.grad_loss_weight,
            "disp": reg_out,
        }

        # Dictionary of results
        align_time = time.time() - start_time
        result_dict["time"] = (align_time,)
        return result_dict

    def forward_loss(self, network_output, img_m, img_f):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        disp = network_output
        disp_permute = disp.permute(0, 2, 3, 4, 1)
        flow = displacement2pytorchflow(disp_permute, input_space="norm")
        flow = flow.float()
        img_a = align_img(flow, img_m)

        reg_loss = F.mse_loss(img_a, img_f)
        grad_loss = loss_ops.Grad(loss_mult=1.0)(None, disp)
        loss = reg_loss + self.grad_loss_weight * grad_loss
        return loss, reg_loss, grad_loss, img_a
