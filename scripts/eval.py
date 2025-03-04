import os
import torch
import torch.nn.functional as F
import numpy as np
import torchio as tio

from actgen.utils import align_img, one_hot
from actgen.viz_tools import imshow_registration_2d, imshow_registration_3d
import actgen.loss_ops as loss_ops

from scripts.script_utils import (
    load_dict_from_json,
    save_dict_as_json,
    parse_test_aug,
)


def run_eval(
    loader,
    registration_model,
    list_of_eval_metrics,
    list_of_eval_names,
    list_of_eval_augs,
    list_of_eval_aligns,
    args,
    save_dir_prefix="eval",
):
    # Check if metrics make sense for the task
    assert len(args.label_names) == 1, "Only one label name supported"
    args.label_name = args.label_names[0]
    if args.label_name in ["group", "sex"]:
        assert "xe" in list_of_eval_metrics, "Must use xe loss"
    if args.label_name in ["age"]:
        assert "mse" in list_of_eval_metrics, "Must use mse loss"

    def _build_metric_dict(names):
        list_of_all_test = []
        for m in list_of_eval_metrics:
            list_of_all_test.append(f"{m}")
        _metrics = {}
        _metrics.update({key: [] for key in list_of_all_test})
        return _metrics

    print(
        list_of_eval_metrics,
        list_of_eval_augs,
        list_of_eval_aligns,
        args.save_dir,
    )

    test_metrics = _build_metric_dict(list_of_eval_names)
    print(test_metrics)

    for i, subject in enumerate(loader):
        if args.early_stop_eval_subjects and i == args.early_stop_eval_subjects:
            break
        for aug in list_of_eval_augs:
            param = parse_test_aug(aug)
            mod1 = subject["modality"][0]
            print(
                f"\n\n\nRunning test: subject {i}/{len(loader)}, mod {mod1}, aug {aug}"
            )

            # Create directory to save images, segs, points, metrics
            mod1_str = "-".join(mod1.split("/")[-2:])
            save_dir = args.model_eval_dir / save_dir_prefix / f"{i}_{mod1_str}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Load metrics (for all alignment types) if they exist, else run registration
            all_metrics_paths = {
                align_type_str: save_dir / f"metrics-{aug}-{align_type_str}.json"
                for align_type_str in list_of_eval_aligns
            }
            if (
                all([os.path.exists(p) for p in all_metrics_paths.values()])
                and args.skip_if_completed
            ):
                print(
                    f"Found metrics for all alignments, skipping running registration..."
                )
                all_metrics = {
                    k: load_dict_from_json(v) for k, v in all_metrics_paths.items()
                }

            else:
                img = subject["img"][tio.DATA]
                label = subject[args.label_name]

                if args.seg_available:
                    (seg,) = subject["seg"][tio.DATA]
                    # One-hot encode segmentations
                    seg_f = one_hot(seg_f)
                    seg_m = one_hot(seg_m)

                # Move to device
                img = img.float().to(args.device)
                label = label.float().to(args.device)
                if args.seg_available:
                    seg = seg.float().to(args.device)

                # Explicitly augment moving image
                # if args.seg_available:
                #     img_m, seg_m = affine_augment(img_m, param, seg=seg_m)
                # else:
                #     img_m = affine_augment(img_m, param)

                with torch.set_grad_enabled(False):
                    res_dict = registration_model(
                        img,
                    )

                # Dictionary to save metrics dictionary for all alignment types
                output = res_dict[args.label_name]

                # if args.visualize:
                #     if args.dim == 2:
                #         imshow_registration_2d(
                #             img_m[0, 0].cpu().detach().numpy(),
                #             img_f[0, 0].cpu().detach().numpy(),
                #             img_a[0, 0].cpu().detach().numpy(),
                #             (
                #                 points_m[0].cpu().detach().numpy()
                #                 if points_m is not None
                #                 else None
                #             ),
                #             (
                #                 points_f[0].cpu().detach().numpy()
                #                 if points_f is not None
                #                 else None
                #             ),
                #             (
                #                 points_a[0].cpu().detach().numpy()
                #                 if points_a is not None
                #                 else None
                #             ),
                #             weights=points_weights,
                #         )
                #         if args.seg_available:
                #             imshow_registration_2d(
                #                 seg_m[0, 0].cpu().detach().numpy(),
                #                 seg_f[0, 0].cpu().detach().numpy(),
                #                 seg_a[0, 0].cpu().detach().numpy(),
                #                 (
                #                     points_m[0].cpu().detach().numpy()
                #                     if points_m is not None
                #                     else None
                #                 ),
                #                 (
                #                     points_f[0].cpu().detach().numpy()
                #                     if points_f is not None
                #                     else None
                #                 ),
                #                 (
                #                     points_a[0].cpu().detach().numpy()
                #                     if points_a is not None
                #                     else None
                #                 ),
                #                 weights=points_weights,
                #             )
                #     else:
                #         imshow_registration_3d(
                #             img_m[0, 0].cpu().detach().numpy(),
                #             img_f[0, 0].cpu().detach().numpy(),
                #             img_a[0, 0].cpu().detach().numpy(),
                #             (
                #                 points_m[0].cpu().detach().numpy()
                #                 if points_m is not None
                #                 else None
                #             ),
                #             (
                #                 points_f[0].cpu().detach().numpy()
                #                 if points_f is not None
                #                 else None
                #             ),
                #             (
                #                 points_a[0].cpu().detach().numpy()
                #                 if points_a is not None
                #                 else None
                #             ),
                #             weights=(
                #                 points_weights[0].cpu().detach().numpy()
                #                 if points_weights is not None
                #                 else None
                #             ),
                #             projection=True,
                #             save_path=None,
                #         )
                #         # imshow_registration_3d(
                #         #     img_m[0, 0].cpu().detach().numpy(),
                #         #     img_f[0, 0].cpu().detach().numpy(),
                #         #     img_a[0, 0].cpu().detach().numpy(),
                #         #     (
                #         #         points_m[0].cpu().detach().numpy()
                #         #         if points_m is not None
                #         #         else None
                #         #     ),
                #         #     (
                #         #         points_f[0].cpu().detach().numpy()
                #         #         if points_f is not None
                #         #         else None
                #         #     ),
                #         #     (
                #         #         points_a[0].cpu().detach().numpy()
                #         #         if points_a is not None
                #         #         else None
                #         #     ),
                #         #     weights=(
                #         #         points_weights[0].cpu().detach().numpy()
                #         #         if points_weights is not None
                #         #         else None
                #         #     ),
                #         #     resize=(256, 256, 256),
                #         #     projection=True,
                #         #     save_path=None,
                #         # )
                #         if args.seg_available:
                #             imshow_registration_3d(
                #                 seg_m.argmax(1)[0].cpu().detach().numpy(),
                #                 seg_f.argmax(1)[0].cpu().detach().numpy(),
                #                 seg_a.argmax(1)[0].cpu().detach().numpy(),
                #                 (
                #                     points_m[0].cpu().detach().numpy()
                #                     if points_m is not None
                #                     else None
                #                 ),
                #                 (
                #                     points_f[0].cpu().detach().numpy()
                #                     if points_f is not None
                #                     else None
                #                 ),
                #                 (
                #                     points_a[0].cpu().detach().numpy()
                #                     if points_a is not None
                #                     else None
                #                 ),
                #                 weights=(
                #                     points_weights[0].cpu().detach().numpy()
                #                     if points_weights is not None
                #                     else None
                #                 ),
                #                 save_path=None,
                #             )

                # Compute metrics
                metrics = {}
                if args.seg_available:
                    # Always compute hard dice once ahead of time
                    dice_total = loss_ops.DiceLoss(hard=True)(
                        seg_a, seg_f, ign_first_ch=True
                    )
                    dice_roi = loss_ops.DiceLoss(hard=True, return_regions=True)(
                        seg_a, seg_f, ign_first_ch=True
                    )
                    dice_total = 1 - dice_total.item()
                    dice_roi = (1 - dice_roi.cpu().detach().numpy()).tolist()
                for m in list_of_eval_metrics:
                    if m == "mse":
                        metrics["mse"] = loss_ops.MSELoss()(output, label).item()
                    elif m == "softdice":
                        assert args.seg_available
                        metrics["softdiceloss"] = loss_ops.DiceLoss()(
                            seg_a, seg_f
                        ).item()
                        metrics["softdice"] = 1 - metrics["softdiceloss"]
                    elif m == "harddice":
                        assert args.seg_available
                        metrics["harddice"] = dice_total
                    elif m == "harddiceroi":
                        assert args.seg_available
                        metrics["harddiceroi"] = dice_roi
                    elif m == "hausd":
                        assert args.seg_available and args.dim == 3
                        metrics["hausd"] = loss_ops.hausdorff_distance(seg_a, seg_f)
                    elif m == "jdstd":
                        assert args.dim == 3
                        if grid is None:
                            metrics["jdstd"] = res_dict["jdstd"]
                        else:
                            grid_permute = grid.permute(0, 4, 1, 2, 3)
                            metrics["jdstd"] = loss_ops.jdstd(grid_permute)
                    elif m == "jdlessthan0":
                        assert args.dim == 3
                        if grid is None:
                            metrics["jdlessthan0"] = res_dict["jdlessthan0"]
                        else:
                            grid_permute = grid.permute(0, 4, 1, 2, 3)
                            metrics["jdlessthan0"] = loss_ops.jdstd(grid_permute)
                    # Classification metrics
                    elif m == "xe":
                        metrics["xe"] = F.cross_entropy(output, label).item()
                    elif m == "acc":
                        output_int = torch.argmax(output, dim=-1)
                        label_int = torch.argmax(label, dim=-1)
                        metrics["acc"] = loss_ops.acc(output_int, label_int)
                    else:
                        raise ValueError('Invalid metric "{}"'.format(m))

                    # Print some stats
                    print("\nDebugging info:")
                    print(f'-> Time: {res_dict["time"]}')
                    print(f"-> Max random params: {param} ")
                    print(f"-> Img shapes: {img.shape}")
                    print(f"-> Float16: {args.use_amp}")
                    if args.seg_available:
                        print(f"-> Seg shapes: {seg_f.shape}, {seg_m.shape}")
                    # print(f"-> Full Results: {res_dict}")

                    print("\nMetrics:")
                    for metric_name, metric in metrics.items():
                        print(f"-> {metric_name}: {metric}")

                    # Save all outputs to disk
                    assert args.batch_size == 1  # TODO: fix this

                    # Save metrics
                    metrics_path = save_dir / f"metrics.json"
                    print("Saving:", metrics_path)
                    save_dict_as_json(metrics, metrics_path)

                    # Save images and grid
                    img_path = save_dir / f"img_f_{i}-{mod1_str}.npy"
                    pred_path = save_dir / f"pred_{i}-{mod1_str}.npy"
                    if not os.path.exists(img_path):
                        print("Saving:", img_path)
                        np.save(img_path, img[0].cpu().detach().numpy())
                    print("Saving:", pred_path)
                    np.save(pred_path, output[0].cpu().detach().numpy())

                    # Save segmentations
                    if args.seg_available:
                        seg_f_path = save_dir / f"seg_f_{i}-{mod1_str}.npy"
                        seg_m_path = save_dir / f"seg_m_{i}-{mod2_str}-{aug}.npy"
                        seg_a_path = (
                            save_dir
                            / f"seg_a_{i}-{mod1_str}-{mod2_str}-{aug}-{align_type_str}.npy"
                        )
                        if not os.path.exists(seg_f_path):
                            print("Saving:", seg_f_path)
                            np.save(
                                seg_f_path,
                                np.argmax(seg_f.cpu().detach().numpy(), axis=1),
                            )
                        if not os.path.exists(seg_m_path):
                            print("Saving:", seg_m_path)
                            np.save(
                                seg_m_path,
                                np.argmax(seg_m.cpu().detach().numpy(), axis=1),
                            )
                        print("Saving:", seg_a_path)
                        np.save(
                            seg_a_path,
                            np.argmax(seg_a.cpu().detach().numpy(), axis=1),
                        )

            # Save metrics in global test_metrics dictionary
            for m in list_of_eval_metrics:
                test_metrics[f"{m}"].append(metrics[m])

    return test_metrics
