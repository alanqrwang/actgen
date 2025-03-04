import pickle
import numpy as np
import os

from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)
import torch
import random

from torch.utils.data import Dataset, DataLoader

# from create_dataset import HCPT1wDataset


def get_t1_all_file_list(
    # file_dir_prefix="/scr/fangruih/stru/",
    file_dir_prefix="/simurgh/u/alanqw/data/fangruih/stru/",
):
    cluster_name = os.getenv("CLUSTER_NAME")
    if cluster_name == "sc":
        prefix = "/simurgh/u/fangruih"
    elif cluster_name == "haic":
        prefix = "/hai/scratch/fangruih"
        file_dir_prefix = "/hai/scratch/fangruih/data/"
    else:
        raise ValueError(
            f"Unknown cluster name: {cluster_name}. Please set the CLUSTER_NAME environment variable correctly."
        )

    dataset_names = [
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/abcd/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/adni_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_ag_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_dev_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/hcp_ya_mpr1/paths_and_info_flexpath.pkl",
        f"{prefix}/monai-tutorials/generative/3d_ldm/metadata/ppmi_t1/paths_and_info_flexpath.pkl",
    ]
    train_images = []
    train_ages = []
    val_images = []
    val_ages = []

    for dataset_name in dataset_names:
        with open(dataset_name, "rb") as file:
            data = pickle.load(file)

            # Convert paths and ages to lists if they are NumPy arrays
            train_new_images = (
                data["train"]["paths"].tolist()
                if isinstance(data["train"]["paths"], np.ndarray)
                else data["train"]["paths"]
            )
            train_new_ages = (
                data["train"]["age"].tolist()
                if isinstance(data["train"]["age"], np.ndarray)
                else data["train"]["age"]
            )
            train_new_sex = (
                data["train"]["sex"].tolist()
                if isinstance(data["train"]["sex"], np.ndarray)
                else data["train"]["sex"]
            )

            val_new_images = (
                data["val"]["paths"].tolist()
                if isinstance(data["val"]["paths"], np.ndarray)
                else data["val"]["paths"]
            )
            val_new_ages = (
                data["val"]["age"].tolist()
                if isinstance(data["val"]["age"], np.ndarray)
                else data["val"]["age"]
            )
            val_new_sex = (
                data["val"]["sex"].tolist()
                if isinstance(data["val"]["sex"], np.ndarray)
                else data["val"]["sex"]
            )

            # Append new data to existing lists
            if not train_images:  # More Pythonic way to check if the list is empty
                # Direct assignment for the first file
                train_images = train_new_images
                train_ages = train_new_ages
                train_sex = train_new_sex

                val_images = val_new_images
                val_ages = val_new_ages
                val_sex = val_new_sex
            else:
                # Concatenation for subsequent files
                train_images += train_new_images
                train_ages += train_new_ages
                train_sex += train_new_sex

                val_images += val_new_images
                val_ages += val_new_ages
                val_sex += val_new_sex

            # Debug output to check the results
            print(train_images[-1])  # Print the last path

    # process z normalization for age

    ages_array = np.array(train_ages)
    # Calculate mean and standard deviation
    age_mean = np.mean(ages_array)
    age_std = np.std(ages_array)
    # train_ages = (ages_array - age_mean) / age_std
    # train_ages = train_ages.tolist()

    # val_ages_array = np.array(val_ages)
    # age_mean = np.mean(val_ages_array)
    # val_ages = (val_ages_array - age_mean) / age_std
    # val_ages = val_ages.tolist()

    train_images = [file_dir_prefix + train_image for train_image in train_images]
    val_images = [file_dir_prefix + val_image for val_image in val_images]

    print(len(train_images))
    print(len(val_images))

    # Zip the conditions into one single list
    train_conditions = [torch.tensor([a, b]) for a, b in zip(train_ages, train_sex)]
    val_conditions = [torch.tensor([a, b]) for a, b in zip(val_ages, val_sex)]

    return train_images, train_conditions, val_images, val_conditions, age_mean, age_std


def prepare_dataloader_from_list(
    args,
    batch_size,
    num_workers,
    shuffle_for_train=True,
    with_conditioning=False,
    cross_attention_dim=None,
    expand_token_times=None,
):
    channel = 0

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
            # EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=args.spacing, mode=("bilinear")),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0, upper=99.5, b_min=0, b_max=1
            ),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )

    train_images, train_conditions, val_images, val_conditions, age_mean, age_std = (
        get_t1_all_file_list()
    )

    train_ds = FileListDataset(
        train_images,
        condition_list=train_conditions,
        transform=train_transforms,
        with_conditioning=with_conditioning,
        cross_attention_dim=cross_attention_dim,
        expand_token_times=expand_token_times,
        compute_dtype=torch.float32,
    )

    val_ds = FileListDataset(
        val_images,
        condition_list=val_conditions,
        transform=val_transforms,
        with_conditioning=with_conditioning,
        cross_attention_dim=cross_attention_dim,
        expand_token_times=expand_token_times,
        compute_dtype=torch.float32,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_for_train,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader  # train_ds , val_ds #


class T1FangruiDataset:
    def __init__(self, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """

        class Args:
            pass

        args = Args()
        args.spacing = (1.0, 1.0, 1.0)
        args.channel = 0

        self.train_loader, self.val_loader = prepare_dataloader_from_list(
            args,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_for_train=shuffle,
            with_conditioning=True,
            cross_attention_dim=256,
            expand_token_times=200,
        )

        # Preprocess the metadata
        # print("Preprocessing...")
        # for subject in self.train_subjects + self.val_subjects + self.test_subjects:
        #     subject["age"] = torch.tensor(subject["age"])[..., None]
        #     subject["sex"] = F.one_hot(torch.tensor(subject["sex"]).long(), 2)
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.shuffle = shuffle
        # self.seg_available = False

    def get_reference_subject(self, transform=None):
        pass

    def get_pretrain_loader(
        self, batch_size, num_workers, transform=None, paired=False
    ):
        return self.train_loader

    def get_train_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the training set.

        Returns:
        - train_loader: DataLoader for the training set
        """
        return self.train_loader

    def get_val_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the validation set.

        Returns:
        - val_loader: DataLoader for the validation set
        """
        return self.val_loader

    def get_test_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the test set.

        Returns:
        - test_loader: DataLoader for the test set
        """
        return self.val_loader


class FileListDataset(Dataset):
    def __init__(
        self,
        file_list,
        condition_list=None,
        with_conditioning=False,
        transform=None,
        cross_attention_dim=None,
        expand_token_times=None,
        compute_dtype=torch.float32,
    ):
        self.file_list = file_list
        self.transform = transform
        self.with_conditioning = with_conditioning
        self.compute_dtype = compute_dtype
        self.cross_attention_dim = cross_attention_dim
        self.expand_token_times = expand_token_times
        self.condition_list = condition_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = {"image": img_path}

        if self.transform:

            data = self.transform(data)

        if self.with_conditioning:
            condition_tensor = self.condition_list[idx]
            data["age"] = condition_tensor[0]
            data["sex"] = condition_tensor[1]

            condition_tensor = condition_tensor.unsqueeze(-1)
            condition_tensor = condition_tensor.expand(-1, self.cross_attention_dim)
            condition_tensor = condition_tensor.repeat(self.expand_token_times, 1)

            data["condition"] = condition_tensor

        return data
