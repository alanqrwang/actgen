import os
import torch
import torch.nn.functional as F
import numpy as np
import torchio as tio
import time
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt

from actgen.utils import align_img, one_hot, one_hot_eval_synthseg
from actgen.viz_tools import (
    imshow_registration_2d,
    imshow_registration_3d,
    imshow_channels,
)
import actgen.loss_ops as loss_ops

from scripts.script_utils import aggregate_dicts


def run_train(train_loader, model, model_ema, optimizer, args):
    """Train for one epoch.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        model: Registration model
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda")

    model.train()

    res = []

    for step_idx, subject in enumerate(train_loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break

        # Get images and segmentations from TorchIO subject
        imgs = []
        if "T1_baseline_ses-01" in subject:
            imgs.append(subject["T1_baseline_ses-01"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T3_6mo_ses-03" in subject:
            imgs.append(subject["T3_6mo_ses-03"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T4_12mo_ses-04" in subject:
            imgs.append(subject["T4_12mo_ses-04"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T5_18mo_ses-05" in subject:
            imgs.append(subject["T5_18mo_ses-05"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        imgs = torch.stack(imgs, dim=1).float().to(args.device)
        label = torch.tensor(subject["group"]).long().to(args.device)
        # print(imgs.shape)

        # Move to device
        # img = img.float().to(args.device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                if args.use_profiler:
                    with profile(
                        enabled=args.use_profiler,
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack=True,
                        profile_memory=True,
                        experimental_config=torch._C._profiler._ExperimentalConfig(
                            verbose=True
                        ),
                    ) as prof:
                        with record_function("model_inference"):
                            model_out = model(imgs)
                    print(
                        prof.key_averages(group_by_stack_n=5).table(
                            sort_by="self_cuda_memory_usage"
                        )
                    )
                else:
                    model_out = model(imgs)
                model_out_ema = model_ema(imgs)
                # print("model out", model_out.shape)
                # print("label", label, label.shape)

                # Compute metrics
                metrics = {}

                # Compute loss
                if args.loss_type == "mse":
                    metrics["train/mse"] = loss_ops.MSELoss()(model_out, label)
                    loss = metrics["train/mse"]
                elif args.loss_type == "xe":
                    metrics["train/xe"] = F.cross_entropy(model_out, label)
                    preds = model_out.argmax(dim=1)
                    metrics["train/acc"] = (preds == label).float().mean()
                    loss = metrics["train/xe"]
                else:
                    raise ValueError(f"Loss type {args.loss_type} not recognized")
                metrics["train/loss"] = loss

        # Perform backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        end_time = time.time()
        metrics["train/epoch_time"] = end_time - start_time

        model_ema.update()

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

        if args.debug_mode:
            print("\nDebugging info:")
            print(f"-> Img shapes: {imgs.shape}")
            print(f"-> Float16: {args.use_amp}")

        # if args.visualize and step_idx == 0:
        #     plt.imshow(img[0, 0, img.shape[2] // 2].cpu().detach().numpy())
        #     plt.title(f"GT: {label}")
        #     if not args.debug_mode:
        #         plt.savefig(
        #             os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png")
        #         )
        #     plt.show()
        #     plt.close()

    return aggregate_dicts(res)


def run_val(val_loader, model, model_ema, args, split_name="val"):
    """Train for one epoch.

    Args:
        train_loader: Dataloader which returns pair of TorchIO subjects per iteration
        model: Registration model
        optimizer: Pytorch optimizer
        args: Other script arguments
    """
    start_time = time.time()

    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda")

    model.train()

    res = []

    for step_idx, subject in enumerate(val_loader):
        if args.steps_per_epoch and step_idx == args.steps_per_epoch:
            break

        # Get images and segmentations from TorchIO subject
        imgs = []
        if "T1_baseline_ses-01" in subject:
            imgs.append(subject["T1_baseline_ses-01"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T3_6mo_ses-03" in subject:
            imgs.append(subject["T3_6mo_ses-03"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T4_12mo_ses-04" in subject:
            imgs.append(subject["T4_12mo_ses-04"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        if "T5_18mo_ses-05" in subject:
            imgs.append(subject["T5_18mo_ses-05"][tio.DATA])
        # else:
        #     imgs.append(torch.zeros((1, 1, 192, 256, 256)))
        imgs = torch.stack(imgs, dim=1).float().to(args.device)
        label = torch.tensor(subject["group"]).long().to(args.device)
        # print(imgs.shape)

        with torch.set_grad_enabled(False):
            with torch.amp.autocast(
                device_type="cuda", enabled=args.use_amp, dtype=torch.float16
            ):
                if args.use_profiler:
                    with profile(
                        enabled=args.use_profiler,
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack=True,
                        profile_memory=True,
                        experimental_config=torch._C._profiler._ExperimentalConfig(
                            verbose=True
                        ),
                    ) as prof:
                        with record_function("model_inference"):
                            model_out = model(imgs)
                    print(
                        prof.key_averages(group_by_stack_n=5).table(
                            sort_by="self_cuda_memory_usage"
                        )
                    )
                else:
                    model_out = model(imgs)
                model_out_ema = model_ema(imgs)
                # print("model out", model_out.shape)
                # print("label", label, label.shape)

                # Compute metrics
                metrics = {}

                # Compute loss
                if args.loss_type == "mse":
                    metrics[f"{split_name}/mse"] = loss_ops.MSELoss()(model_out, label)
                    loss = metrics[f"{split_name}/mse"]
                elif args.loss_type == "xe":
                    metrics[f"{split_name}/xe"] = F.cross_entropy(model_out, label)
                    preds = model_out.argmax(dim=1)
                    metrics[f"{split_name}/acc"] = (preds == label).float().mean()
                    loss = metrics[f"{split_name}/xe"]
                else:
                    raise ValueError(f"Loss type {args.loss_type} not recognized")
                metrics[f"{split_name}/loss"] = loss

        end_time = time.time()
        metrics[f"{split_name}/epoch_time"] = end_time - start_time

        model_ema.update()

        # Convert metrics to numpy
        metrics = {
            k: torch.as_tensor(v).detach().cpu().numpy().item()
            for k, v in metrics.items()
        }
        res.append(metrics)

        if args.debug_mode:
            print("\nDebugging info:")
            print(f"-> Img shapes: {imgs.shape}")
            print(f"-> Float16: {args.use_amp}")

        # if args.visualize and step_idx == 0:
        #     plt.imshow(img[0, 0, img.shape[2] // 2].cpu().detach().numpy())
        #     plt.title(f"GT: {label}")
        #     if not args.debug_mode:
        #         plt.savefig(
        #             os.path.join(args.model_img_dir, f"img_{args.curr_epoch}.png")
        #         )
        #     plt.show()
        #     plt.close()

    return aggregate_dicts(res)
