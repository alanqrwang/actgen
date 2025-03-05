import json
import os
import random
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchio as tio
from ema_pytorch import EMA

import wandb
from dataset import (
    act_dataset,
)
from actgen import utils as keymorph_utils
from scripts import script_utils
from scripts.eval import run_eval
from scripts.script_utils import (
    ParseKwargs,
    initialize_wandb,
    save_dict_as_json,
    define_instance,
)
from scripts.train import run_train, run_val
from stai_utils.datasets.dataset_utils import T1All


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )

    args = parser.parse_args()

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    return args


def create_dirs(args):
    arg_dict = vars(deepcopy(args))

    # Add run_mode prefix to run name
    run_name_prefix = f"__{args.run_mode}__"
    args.run_name = run_name_prefix + args.run_name

    # Path to save outputs
    arguments = (
        args.run_name
        + "_data"
        + str(args.dataset_type)
        + "_batch"
        + str(args.batch_size)
        + "_lr"
        + str(args.lr)
        + "_seed"
        + str(args.seed)
    )

    args.model_dir = Path(args.save_dir) / arguments
    if not os.path.exists(args.model_dir) and not args.debug_mode:
        print("Creating directory: {}".format(args.model_dir))
        os.makedirs(args.model_dir)

    if args.run_mode == "eval":
        args.model_eval_dir = args.model_dir / "eval"
        if not os.path.exists(args.model_eval_dir) and not args.debug_mode:
            os.makedirs(args.model_eval_dir)

    else:
        args.model_ckpt_dir = args.model_dir / "checkpoints"
        if not os.path.exists(args.model_ckpt_dir) and not args.debug_mode:
            os.makedirs(args.model_ckpt_dir)
        args.model_img_dir = args.model_dir / "train_img"
        if not os.path.exists(args.model_img_dir) and not args.debug_mode:
            os.makedirs(args.model_img_dir)

    # Write arguments to json
    if not args.debug_mode:
        with open(os.path.join(args.model_dir, "args.json"), "w") as outfile:
            json.dump(arg_dict, outfile, sort_keys=True, indent=4)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_data(args):
    if args.dataset_type == "act":
        transform = tio.Compose(
            [
                tio.ToCanonical(),
                tio.Resample("T1_baseline_ses-01"),
                # tio.Resample((1, 1, 1)),
                # tio.CropOrPad(
                #     (192, 256, 256),
                #     padding_mode=0,
                #     include=(
                #         "T1_baseline_ses-01",
                #         "T3_6mo_ses-03",
                #         "T4_12mo_ses-04",
                #         "T5_18mo_ses-05",
                #     ),
                # ),
                # tio.Lambda(keymorph_utils.rescale_intensity, include=("img",)),
            ]
        )
        dataset = act_dataset.ACTDataset(args.data_path, args.demo_csv)
    else:
        raise ValueError('Invalid dataset "{}"'.format(args.dataset_type))

    return {
        "transform": transform,
        "train": dataset.get_train_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
        "val": dataset.get_val_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
        "test": dataset.get_test_loader(
            args.batch_size, args.num_workers, transform=transform
        ),
    }


def get_model(args):
    model = define_instance(args, "model_def")

    model_ema = EMA(
        model,
        beta=0.9999,  # exponential moving average factor
        update_after_step=1,  # only after this number of .update() calls will it start updating
        update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )
    model.to(args.device)
    model_ema.to(args.device)
    script_utils.summary(model)

    return model, model_ema


def main():
    args = parse_args()
    if args.debug_mode:
        args.steps_per_epoch = 3
        args.early_stop_eval_subjects = 1
    pprint(vars(args))

    # Create run directories
    create_dirs(args)

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        print("WARNING! No GPU available, using the CPU instead...")
        args.device = torch.device("cpu")
    print("Number of GPUs: {}".format(torch.cuda.device_count()))

    # Set seed
    set_seed(args)

    # Data
    loaders = get_data(args)

    # Model
    model, model_ema = get_model(args)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Checkpoint loading
    args.resume = False
    if args.resume_latest:
        args.resume = True
        args.load_path = script_utils.get_latest_epoch_file(args.model_ckpt_dir, args)
        if args.load_path is None:
            raise ValueError(
                f"No checkpoint found to resume from: {args.model_ckpt_dir}"
            )
    if args.load_path is not None:
        print(f"Loading checkpoint from {args.load_path}")
        ckpt_state, model, optimizer = script_utils.load_checkpoint(
            args.load_path,
            model,
            optimizer,
            device=args.device,
        )

    if args.run_mode == "eval":
        pass

    else:
        model.train()

        if args.use_wandb and not args.debug_mode:
            initialize_wandb(args)

        if args.resume:
            start_epoch = ckpt_state["epoch"] + 1
        else:
            start_epoch = 1

        for epoch in range(start_epoch, args.n_epochs + 1):
            args.curr_epoch = epoch
            print(f"\nEpoch {epoch}/{args.n_epochs}")
            train_epoch_stats = run_train(
                loaders["train"],
                model,
                model_ema,
                optimizer,
                args,
            )

            for metric_name, metric in train_epoch_stats.items():
                print(f"[Train Stat] {metric_name}: {metric:.5f}")

            val_epoch_stats = run_val(
                loaders["val"], model, model_ema, args, split_name="val"
            )

            for metric_name, metric in val_epoch_stats.items():
                print(f"[Val Stat] {metric_name}: {metric:.5f}")

            test_epoch_stats = run_val(
                loaders["test"], model, model_ema, args, split_name="test"
            )

            for metric_name, metric in test_epoch_stats.items():
                print(f"[Test Stat] {metric_name}: {metric:.5f}")

            if args.use_wandb and not args.debug_mode:
                epoch_stats = {
                    **train_epoch_stats,
                    **val_epoch_stats,
                    **test_epoch_stats,
                }
                wandb.log(epoch_stats)

            # Save model
            if epoch % args.log_interval == 0 and not args.debug_mode:
                state = {
                    "epoch": epoch,
                    "args": args,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    state,
                    os.path.join(
                        args.model_ckpt_dir,
                        "epoch{}_trained_model.pth.tar".format(epoch),
                    ),
                )


if __name__ == "__main__":
    main()
