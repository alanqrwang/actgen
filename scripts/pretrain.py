import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from ema_pytorch import EMA

from rssl import loss_ops
from rssl.augmentation import random_affine_augment, random_affine_augment_pair
from rssl.utils import align_img, displacement2pytorchflow, random_masking, one_hot
from rssl.viz_tools import (
    imshow_registration_2d,
    imshow_registration_3d,
    imshow_channels,
    imshow_img_and_points_3d,
)
from scripts.script_utils import aggregate_dicts


def run_pretrain_ssl(single_loader, model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    if args.use_ema_encoder:
        model_student = model
        model_teacher = EMA(
            model,
            beta=0.996,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    else:
        model_student = model
        model_teacher = model

    max_random_params = args.max_random_affine_augment_params
    res = []

    for step_idx, subject in enumerate(single_loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break
        img = subject["img"][tio.DATA]

        # Move to device
        img = img.float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augment fixed image to get moving image and points
        img_m = random_affine_augment(
            img,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )
        img_f = random_affine_augment(
            img,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            model_outputs_s = model_student(img_m)
            model_outputs_t = model_teacher(img_f)

            feat_s = model_outputs_s["enc_out"].float()
            feat_t = model_outputs_t["enc_out"].float()
            vec_s = model_outputs_s["avgpool_out"].float()
            vec_t = model_outputs_t["avgpool_out"].float()
            proj_s = model_outputs_s["proj_lin"].float()
            proj_t = model_outputs_t["proj_lin"].float()

            if args.pretrain_loss_fn == "byol":
                # L2 normalize
                proj_s_norm = proj_s / proj_s.norm(p=2, dim=1, keepdim=True)
                vec_t_norm = vec_t / vec_t.norm(p=2, dim=1, keepdim=True)
                loss1 = loss_ops.MSELoss()(proj_s_norm, vec_t_norm.detach())
                loss = loss1
            elif args.pretrain_loss_fn == "dino":
                loss1 = loss_ops.DINOLoss(out_dim=256)(vec_s, vec_t.detach())
                loss = loss1
            else:
                raise ValueError("Invalid loss function")

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update teacher model
        model_teacher.update()

        # Compute metrics
        metrics = {}
        metrics["pretrain/scale_augment"] = scale_augment
        metrics["pretrain/loss"] = loss.cpu().detach().numpy()
        # metrics["pretrain/reg_consistency_loss_m2f"] = (
        #     reg_consistency_loss_m2f.cpu().detach().numpy()
        # )
        # metrics["pretrain/reg_consistency_loss_f2m"] = (
        #     reg_consistency_loss_f2m.cpu().detach().numpy()
        # )
        # metrics["pretrain/cov_loss_m"] = cov_loss_m.cpu().detach().numpy()
        # metrics["pretrain/cov_loss_f"] = cov_loss_f.cpu().detach().numpy()
        # metrics["pretrain/harddiceloss"] = loss_ops.DiceLoss(hard=True)(
        #     seg_pred.cpu().detach(), seg_m.cpu().detach()
        # )
        # metrics["pretrain/harddice"] = 1 - metrics["pretrain/harddiceloss"]
        end_time = time.time()
        metrics["pretrain/epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            imshow_channels(
                feat_s[0].cpu().detach().numpy(),
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"feat_s_{args.curr_epoch}.png"
                    )
                ),
            )
            imshow_channels(
                feat_t[0].cpu().detach().numpy(),
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"feat_t_{args.curr_epoch}.png"
                    )
                ),
            )

            imshow_registration_3d(
                img_m[0, 0].cpu().detach().numpy(),
                img_f[0, 0].cpu().detach().numpy(),
                img_f[0, 0].cpu().detach().numpy(),
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png")
                ),
            )

    return aggregate_dicts(res)


def run_pretrain_rssldecode(loader, model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    assert not args.use_ema_encoder, "Don't use ema encoder"

    max_random_params = args.max_random_affine_augment_params
    max_random_params_pair = args.max_random_affine_augment_params_pair
    res = []

    for step_idx, subject in enumerate(loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break
        if isinstance(subject, tuple) and len(subject) == 2:
            fixed, moving = subject
        else:
            fixed = subject
            moving = next(iter(loader))

        if "img" in fixed:
            img_f = subject["img"][tio.DATA]
        else:
            img_f = subject["image"]
        if "img" in moving:
            img_m = subject["img"][tio.DATA]
        else:
            img_m = subject["image"]

        # Move to device
        img_m = img_m.float().to(args.device)
        img_f = img_f.float().to(args.device)
        # label_m = label_m.float().to(args.device)
        # label_f = label_f.float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augmentations
        img_m, img_f = random_affine_augment_pair(
            img_m,
            img_f,
            max_random_params=max_random_params_pair,
            scale_params=scale_augment,
        )

        img_m = random_affine_augment(
            img_m,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        img_f = random_affine_augment(
            img_f,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        # Perform random masking (on fixed image only!!) a la MAEs
        orig_img_f = img_f.clone()
        orig_img_m = img_m.clone()
        if args.mask_ratio > 0:
            img_f, mask_f = random_masking(
                img_f, mask_ratio=args.mask_ratio, patch_size=(16, 16, 16), mask_val=0
            )
            img_m, mask_m = random_masking(
                img_m, mask_ratio=args.mask_ratio, patch_size=(16, 16, 16), mask_val=0
            )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                model_outputs = model(img_m, img_f)

                feat_m = model_outputs["enc_out_m"]
                feat_f = model_outputs["enc_out_f"]

                if args.pretrain_loss_fn == "mse+grad":
                    disp = model_outputs["reg_out"]
                    disp_permute = disp.permute(0, 2, 3, 4, 1)
                    flow = displacement2pytorchflow(disp_permute, input_space="norm")
                    flow = flow.float()
                    img_a = align_img(flow, orig_img_m)
                    reg_loss = loss_ops.MSELoss()(img_a, orig_img_f)
                    grad_loss = loss_ops.Grad(loss_mult=args.grad_loss_weight)(
                        None, disp
                    )
                    seg_loss = torch.tensor(0.0)
                    loss = reg_loss + grad_loss
                elif args.pretrain_loss_fn == "dice":
                    seg_m = one_hot(
                        moving["seg"][tio.DATA].long().to(args.device),
                        num_classes=15,
                    ).float()
                    seg_pred = model_outputs["seg_out"]
                    seg_pred = F.softmax(seg_pred, dim=1)
                    seg_loss = loss_ops.DiceLoss()(seg_pred, seg_m)
                    reg_loss = torch.tensor(0.0)
                    grad_loss = torch.tensor(0.0)
                    loss = seg_loss
                elif args.pretrain_loss_fn == "dice+mse+grad":
                    seg_m = one_hot(
                        moving["seg"][tio.DATA].long().to(args.device),
                        num_classes=15,
                    ).float()
                    disp = model_outputs["reg_out"]
                    disp_permute = disp.permute(0, 2, 3, 4, 1)
                    flow = displacement2pytorchflow(disp_permute, input_space="norm")
                    flow = flow.float()
                    img_a = align_img(flow, img_m)
                    # Only compute loss on unmasked regions
                    if args.mask_ratio > 0:
                        reg_loss = loss_ops.masked_mse_loss(img_a, img_f, mask=mask_f)
                    else:
                        reg_loss = loss_ops.MSELoss()(img_a, img_f)
                    grad_loss = loss_ops.Grad(loss_mult=args.grad_loss_weight)(
                        None, disp
                    )
                    seg_pred = model_outputs["seg_out"]
                    seg_pred = F.softmax(seg_pred, dim=1)
                    seg_loss = loss_ops.DiceLoss()(seg_pred, seg_m)
                    loss = reg_loss + grad_loss + seg_loss
                elif args.pretrain_loss_fn == "two_decode":
                    alpha = min(1, args.curr_epoch / args.anneal_alpha)
                    # alpha = 0
                    img_f_recon = model_outputs["recon_out"]
                    recon_loss = loss_ops.MSELoss()(img_f_recon, img_f)
                    disp = model_outputs["reg_out"]
                    disp_permute = disp.permute(0, 2, 3, 4, 1)
                    flow = displacement2pytorchflow(disp_permute, input_space="norm")
                    flow = flow.float()
                    img_a = align_img(flow, img_m)
                    # Only compute loss on unmasked regions
                    if args.mask_ratio > 0:
                        reg_loss = loss_ops.masked_mse_loss(
                            img_a, img_f_recon, mask=mask_f
                        )
                    else:
                        reg_loss = loss_ops.MSELoss()(img_a, img_f_recon)
                    grad_loss = loss_ops.Grad(loss_mult=args.grad_loss_weight)(
                        None, disp
                    )
                    loss = alpha * (reg_loss + grad_loss) + (1 - alpha) * recon_loss
                else:
                    raise ValueError("Invalid loss function")

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics["pretrain/scale_augment"] = scale_augment
        metrics["pretrain/loss"] = loss.item()
        # metrics["pretrain/seg_loss"] = seg_loss.item()
        metrics["pretrain/reg_loss"] = reg_loss.item()
        metrics["pretrain/grad_loss"] = grad_loss.item()
        # metrics["pretrain/recon_loss"] = recon_loss.item()
        # metrics["pretrain/alpha"] = alpha
        # metrics["pretrain/reg_consistency_loss_m2f"] = (
        #     reg_consistency_loss_m2f.cpu().detach().numpy()
        # )
        # metrics["pretrain/reg_consistency_loss_f2m"] = (
        #     reg_consistency_loss_f2m.cpu().detach().numpy()
        # )
        # metrics["pretrain/cov_loss_m"] = cov_loss_m.cpu().detach().numpy()
        # metrics["pretrain/cov_loss_f"] = cov_loss_f.cpu().detach().numpy()
        # metrics["pretrain/harddiceloss"] = loss_ops.DiceLoss(hard=True)(
        #     seg_pred.cpu().detach(), seg_m.cpu().detach()
        # )
        # metrics["pretrain/harddice"] = 1 - metrics["pretrain/harddiceloss"]
        end_time = time.time()
        metrics["pretrain/epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            points_m = None
            points_f = None
            points_a = None

            if "mse+grad" in args.pretrain_loss_fn:
                imshow_registration_3d(
                    orig_img_m[0, 0].cpu().detach().numpy(),
                    orig_img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"origimg_{args.curr_epoch}.png"
                        )
                    ),
                )
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )

                imshow_img_and_points_3d(
                    img=flow[0, ..., 0].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"flow0_{args.curr_epoch}.png"
                        )
                    ),
                )
                imshow_img_and_points_3d(
                    img=flow[0, ..., 1].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"flow1_{args.curr_epoch}.png"
                        )
                    ),
                )
                imshow_img_and_points_3d(
                    img=flow[0, ..., 2].cpu().detach().numpy(),
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"flow2_{args.curr_epoch}.png"
                        )
                    ),
                )

            if "dice" in args.pretrain_loss_fn:
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    seg_pred[0].argmax(0).cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"seg_{args.curr_epoch}.png"
                        )
                    ),
                )

            if "two_decode" in args.pretrain_loss_fn:
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_a[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"img_{args.curr_epoch}.png"
                        )
                    ),
                )

                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_f_recon[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"recon_{args.curr_epoch}.png"
                        )
                    ),
                )

            # imshow_channels(
            #     feat_m[0].cpu().detach().numpy(),
            #     save_path=(
            #         None
            #         if args.debug_mode
            #         else os.path.join(
            #             args.model_img_dir, f"feat_m_{args.curr_epoch}.png"
            #         )
            #     ),
            # )
            # imshow_channels(
            #     feat_f[0].cpu().detach().numpy(),
            #     save_path=(
            #         None
            #         if args.debug_mode
            #         else os.path.join(
            #             args.model_img_dir, f"feat_f_{args.curr_epoch}.png"
            #         )
            #     ),
            # )

    return aggregate_dicts(res)


def run_pretrain_recon(loader, model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    assert not args.use_ema_encoder, "Don't use ema encoder"

    max_random_params = args.max_random_affine_augment_params
    max_random_params_pair = args.max_random_affine_augment_params_pair
    res = []

    for step_idx, subject in enumerate(loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break
        if "img" in subject:
            img = subject["img"][tio.DATA]
        else:
            img = subject["image"]

        # Move to device
        img = img.float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augmentations
        img = random_affine_augment(
            img,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                model_outputs = model(img)

                feat = model_outputs["enc_out"]

                img_recon = model_outputs["recon_out"]
                recon_loss = loss_ops.MSELoss()(img_recon, img)
                loss = recon_loss

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics["pretrain/scale_augment"] = scale_augment
        metrics["pretrain/loss"] = loss.item()
        end_time = time.time()
        metrics["pretrain/epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            points_m = None
            points_f = None
            points_a = None

            imshow_registration_3d(
                img[0, 0].cpu().detach().numpy(),
                img[0, 0].cpu().detach().numpy(),
                img_recon[0, 0].cpu().detach().numpy(),
                points_m,
                points_f,
                points_a,
                projection=True,
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"recon_{args.curr_epoch}.png"
                    )
                ),
            )

            # imshow_channels(
            #     feat[0].cpu().detach().numpy(),
            #     save_path=(
            #         None
            #         if args.debug_mode
            #         else os.path.join(
            #             args.model_img_dir, f"feat_m_{args.curr_epoch}.png"
            #         )
            #     ),
            # )
    return aggregate_dicts(res)


def run_pretrain_mae(loader, model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    assert not args.use_ema_encoder, "Don't use ema encoder"

    max_random_params = args.max_random_affine_augment_params
    max_random_params_pair = args.max_random_affine_augment_params_pair
    res = []

    for step_idx, subject in enumerate(loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break

        if args.pretrain_type == "rsslmae":
            if isinstance(subject, tuple) and len(subject) == 2:
                fixed, moving = subject
            else:
                fixed = subject
                moving = next(iter(loader))
        else:
            fixed = subject
            moving = None

        if "img" in fixed:
            img_f = fixed["img"][tio.DATA]
        else:
            img_f = fixed["image"]

        if args.pretrain_type == "rsslmae":
            if "img" in moving:
                img_m = moving["img"][tio.DATA]
            else:
                img_m = moving["image"]

        # Move to device
        img_f = img_f.float().to(args.device)
        if args.pretrain_type == "rsslmae":
            img_m = img_m.float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augmentations
        img_f = random_affine_augment(
            img_f,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )
        if args.pretrain_type == "rsslmae":
            img_m = random_affine_augment(
                img_m,
                max_random_params=max_random_params,
                scale_params=scale_augment,
            )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                if args.pretrain_type == "rsslmae":
                    model_outputs = model(img_m, img_f)
                else:
                    model_outputs = model(img_f)

                loss = model_outputs["loss"]
                img_recon = model_outputs["recon_out"]

                # img_recon = model_outputs["recon_out"]
                # recon_loss = loss_ops.MSELoss()(img_recon, img)
                # loss = recon_loss

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics["pretrain/scale_augment"] = scale_augment
        metrics["pretrain/loss"] = loss.item()
        if args.pretrain_type == "rsslmae":
            metrics["pretrain/grad_loss"] = model_outputs["grad_loss"].item()
            metrics["pretrain/reg_loss"] = model_outputs["reg_loss"].item()
            metrics["pretrain/grad_loss_weight"] = model_outputs["grad_loss_weight"]
        end_time = time.time()
        metrics["pretrain/epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            points_m = None
            points_f = None
            points_a = None

            if args.pretrain_type == "rsslmae":
                imshow_registration_3d(
                    img_m[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_recon[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"recon_{args.curr_epoch}.png"
                        )
                    ),
                )
            else:
                imshow_registration_3d(
                    img_f[0, 0].cpu().detach().numpy(),
                    img_f[0, 0].cpu().detach().numpy(),
                    img_recon[0, 0].cpu().detach().numpy(),
                    points_m,
                    points_f,
                    points_a,
                    projection=True,
                    save_path=(
                        None
                        if args.debug_mode
                        else os.path.join(
                            args.model_img_dir, f"recon_{args.curr_epoch}.png"
                        )
                    ),
                )

            # imshow_channels(
            #     feat[0].cpu().detach().numpy(),
            #     save_path=(
            #         None
            #         if args.debug_mode
            #         else os.path.join(
            #             args.model_img_dir, f"feat_m_{args.curr_epoch}.png"
            #         )
            #     ),
            # )
    return aggregate_dicts(res)


def run_pretrain_multirsslmae(loader, model, optimizer, args):
    """Run pretraining loop for a single epoch.

    Given reference points, pretraining samples a random subject in the training set and applies an
    affine augmentation to the subject's image and the reference points. The model then takes as input
    the augmented image and tries to predict the corresponding augmented reference points.
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()

    assert not args.use_ema_encoder, "Don't use ema encoder"

    max_random_params = args.max_random_affine_augment_params
    max_random_params_pair = args.max_random_affine_augment_params_pair
    res = []

    for step_idx, subject in enumerate(loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break

        if isinstance(subject, tuple) and len(subject) == 2:
            fixed, moving = subject
        else:
            fixed = subject
            moving = next(iter(loader))

        if "img" in fixed:
            img_f = fixed["img"][tio.DATA]
        else:
            img_f = fixed["image"]

        if "img" in moving:
            img_m = moving["img"][tio.DATA]
        else:
            img_m = moving["image"]

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)

        # Deform image and fixed points
        if args.affine_slope >= 0:
            scale_augment = np.clip(args.curr_epoch / args.affine_slope, None, 1)
        else:
            scale_augment = 1

        # Affine augmentations
        img_f = random_affine_augment(
            img_f,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )
        img_m = random_affine_augment(
            img_m,
            max_random_params=max_random_params,
            scale_params=scale_augment,
        )

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                model_outputs = model(img_m, img_f)

                loss = model_outputs["loss"]

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics["pretrain/scale_augment"] = scale_augment
        metrics["pretrain/loss"] = loss.item()
        metrics["pretrain/reg_loss"] = model_outputs["reg_loss"].item()
        metrics["pretrain/sim_loss"] = model_outputs["sim_loss"].item()
        metrics["pretrain/grad_loss"] = model_outputs["grad_loss"].item()
        metrics["pretrain/recon_loss"] = model_outputs["recon_loss"].item()
        metrics["pretrain/grad_loss_weight"] = model_outputs["grad_loss_weight"]
        end_time = time.time()
        metrics["pretrain/epoch_time"] = end_time - start_time
        res.append(metrics)

        if args.visualize and step_idx == 0:
            img_reg = model_outputs["reg_out"]
            img_recon_m = model_outputs["recon_out_m"]
            img_recon_f = model_outputs["recon_out_f"]
            imshow_registration_3d(
                img_m[0, 0].cpu().detach().numpy(),
                img_f[0, 0].cpu().detach().numpy(),
                img_reg[0, 0].cpu().detach().numpy(),
                projection=True,
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"recon_{args.curr_epoch}.png"
                    )
                ),
            )
            imshow_registration_3d(
                img_m[0, 0].cpu().detach().numpy(),
                img_m[0, 0].cpu().detach().numpy(),
                img_recon_m[0, 0].cpu().detach().numpy(),
                projection=True,
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"recon_m_{args.curr_epoch}.png"
                    )
                ),
            )
            imshow_registration_3d(
                img_f[0, 0].cpu().detach().numpy(),
                img_f[0, 0].cpu().detach().numpy(),
                img_recon_f[0, 0].cpu().detach().numpy(),
                projection=True,
                save_path=(
                    None
                    if args.debug_mode
                    else os.path.join(
                        args.model_img_dir, f"recon_f_{args.curr_epoch}.png"
                    )
                ),
            )
    return aggregate_dicts(res)


def run_validation_mae(loader, model, args):
    """
    Run validation loop for a single epoch.

    The validation loop evaluates the model's performance on a validation set by calculating the loss
    and other relevant metrics, such as reconstruction quality.
    """
    model.eval()  # Set model to evaluation mode
    start_time = time.time()

    max_random_params = args.max_random_affine_augment_params
    max_random_params_pair = args.max_random_affine_augment_params_pair
    res = []

    with torch.no_grad():  # Disable gradient calculations
        for step_idx, subject in enumerate(loader):
            if args.steps_per_epoch and step_idx == args.val_steps_per_epoch:
                break

            if args.pretrain_type == "rsslmae":
                if isinstance(subject, tuple) and len(subject) == 2:
                    fixed, moving = subject
                else:
                    fixed = subject
                    moving = next(iter(loader))
            else:
                fixed = subject
                moving = None

            if "img" in fixed:
                img_f = fixed["img"][tio.DATA]
            else:
                img_f = fixed["image"]

            if args.pretrain_type == "rsslmae":
                if "img" in moving:
                    img_m = moving["img"][tio.DATA]
                else:
                    img_m = moving["image"]

            # Move to device
            img_f = img_f.float().to(args.device)
            if args.pretrain_type == "rsslmae":
                img_m = img_m.float().to(args.device)

            # Affine augmentations (applied with validation-specific scaling)
            scale_augment = 1  # No progressive scaling for validation
            img_f = random_affine_augment(
                img_f,
                max_random_params=max_random_params,
                scale_params=scale_augment,
            )
            if args.pretrain_type == "rsslmae":
                img_m = random_affine_augment(
                    img_m,
                    max_random_params=max_random_params,
                    scale_params=scale_augment,
                )

            # Forward pass
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                if args.pretrain_type == "rsslmae":
                    model_outputs = model(img_m, img_f)
                else:
                    model_outputs = model(img_f)

                loss = model_outputs["loss"]
                img_recon = model_outputs["recon_out"]

            # Record metrics
            metrics = {}
            metrics["val/loss"] = loss.item()
            end_time = time.time()
            metrics["val/epoch_time"] = end_time - start_time
            res.append(metrics)

            # Optional visualization
            if args.visualize and step_idx == 0:
                points_m = None
                points_f = None
                points_a = None

                if args.pretrain_type == "rsslmae":
                    imshow_registration_3d(
                        img_m[0, 0].cpu().detach().numpy(),
                        img_f[0, 0].cpu().detach().numpy(),
                        img_recon[0, 0].cpu().detach().numpy(),
                        points_m,
                        points_f,
                        points_a,
                        projection=True,
                        save_path=(
                            None
                            if args.debug_mode
                            else os.path.join(
                                args.model_img_dir, f"val_recon_{args.curr_epoch}.png"
                            )
                        ),
                    )
                else:
                    imshow_registration_3d(
                        img_f[0, 0].cpu().detach().numpy(),
                        img_f[0, 0].cpu().detach().numpy(),
                        img_recon[0, 0].cpu().detach().numpy(),
                        points_m,
                        points_f,
                        points_a,
                        projection=True,
                        save_path=(
                            None
                            if args.debug_mode
                            else os.path.join(
                                args.model_img_dir, f"val_recon_{args.curr_epoch}.png"
                            )
                        ),
                    )

    # Aggregate results from the validation set
    return aggregate_dicts(res)
