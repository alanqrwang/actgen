import os
import torchio as tio
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import itertools
from .utils import PairedDataset
from glob import glob
import pickle


def adnigroup2int(group):
    mapping = {"Patient": 0, "EMCI": 1, "CN": 2, "MCI": 3, "LMCI": 4, "SMC": 5, "AD": 6}
    return torch.tensor(mapping[group]).long()


def adnisex2int(sex):
    mapping = {"M": 0, "F": 1}
    return torch.tensor(mapping[sex]).long()


def create_train_subjects(img_root_path, seg_root_path):
    # Load metadata from CSV
    img_paths = [p for p in os.listdir(img_root_path) if p.endswith(".nii.gz")]
    seg_paths = [os.path.basename(p).replace("img", "label") for p in img_paths]
    subjects = []
    for img_path, seg_path in zip(img_paths, seg_paths):
        subjects.append(
            tio.Subject(
                {
                    "img": tio.ScalarImage(os.path.join(img_root_path, img_path)),
                    "seg": tio.LabelMap(os.path.join(seg_root_path, seg_path)),
                }
            )
        )

    print(f"Total Synapse subjects: {len(subjects)}")
    return subjects


def numpy_reader(path):
    data = np.load(path).astype(np.float32)
    affine = np.eye(4)
    return data, affine


class SynapseDataset:
    def __init__(self, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        root_path = "/simurgh/u/alanqw/data/Synapse_dataset/Abdomen/Abdomen/RawData/Training/img"
        root_path_seg = "/simurgh/u/alanqw/data/Synapse_dataset/Abdomen/Abdomen/RawData/Training/label"
        self.subjects = create_train_subjects(root_path, root_path_seg)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seg_available = True

        # Split the dataset into training, validation, and test sets
        self.train_subjects, self.val_subjects, self.test_subjects = (
            self.split_dataset()
        )

    def split_dataset(self):
        """
        Split the dataset into training, validation, and test sets.

        Returns:
        - train_subjects: list of subjects for training
        - val_subjects: list of subjects for validation
        - test_subjects: list of subjects for testing
        """

        return self.subjects, self.subjects, self.subjects

    def get_reference_subject(self, transform=None):
        return {"0000": self.train_subjects[0]}

    def get_pretrain_loader(
        self, batch_size, num_workers, transform=None, paired=False
    ):
        return self.get_train_loader(
            batch_size, num_workers, transform=transform, paired=paired
        )

    def get_train_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the training set.

        Returns:
        - train_loader: DataLoader for the training set
        """
        if paired:
            train_pairs = list(
                itertools.product(self.train_subjects, self.train_subjects)
            )
            train_dataset = PairedDataset(train_pairs, transform=transform)
        else:
            train_dataset = tio.SubjectsDataset(
                self.train_subjects, transform=transform
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.shuffle,
        )
        return train_loader

    def get_val_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the validation set.

        Returns:
        - val_loader: DataLoader for the validation set
        """
        val_dataset = tio.SubjectsDataset(self.val_subjects, transform=transform)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return val_loader

    def get_test_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the test set.

        Returns:
        - test_loader: DataLoader for the test set
        """
        test_dataset = tio.SubjectsDataset(self.test_subjects, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return test_loader


# Example usage
# root_path = "/scratch/groups/eadeli/data/stru/t1/adni"
# metadata_csv_path = "/scratch/groups/eadeli/data/stru/t1/metadata/adni/preprocessed_ADNI_mri_2_14_2024.csv"
# subjects = create_subjects_from_metadata(root_path, metadata_csv_path)

# # Display the number of subjects created
# print(len(subjects))
