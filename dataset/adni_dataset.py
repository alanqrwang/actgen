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
from collections import defaultdict
import random


# def adnigroup2int(group):
#     mapping = {"Patient": 0, "EMCI": 1, "CN": 2, "MCI": 3, "LMCI": 4, "SMC": 5, "AD": 6}
#     return torch.tensor(mapping[group]).long()


def adnigroup2int(group):
    mapping = {"EMCI": 1, "CN": 0, "MCI": 1, "LMCI": 1, "AD": 2}
    return mapping[group]


def adnisex2int(sex):
    mapping = {"M": 0, "F": 1}
    return mapping[sex]


# def create_subjects_from_metadata(root_path, metadata_csv_path):
#     # Load metadata from CSV
#     metadata_df = pd.read_csv(metadata_csv_path)

#     # Create a list to hold torchio.Subject objects
#     subjects = []

#     # First, check if subjects_list already exists
#     if os.path.exists("subjects_list.pkl"):
#         print("Found subjects_list.pkl!")
#         with open("subjects_list.pkl", "rb") as f:
#             subjects = pickle.load(f)

#     else:
#         # Iterate over the metadata DataFrame
#         print("Didn't find subjects_list.pkl. Creating subjects...")
#         for i, row in metadata_df.iterrows():
#             print(f"{i}/{len(metadata_df)}")
#             image_data_id = row["Image Data ID"]
#             subject_id = row["Subject"]

#             # Construct the expected directory structure based on the given format
#             subject_dir = os.path.join(
#                 root_path, subject_id, "*", "*", image_data_id, "*.npy"
#             )
#             glob_paths = glob(subject_dir)

#             # Walk through the directory structure to find the .npy file
#             for path in glob_paths:
#                 # Create torchio.Subject with metadata
#                 subject = tio.Subject(
#                     img=tio.ScalarImage(path, reader=numpy_reader),
#                     image_data_id=image_data_id,
#                     subject=subject_id,
#                     group=row["Group"],
#                     sex=row["Sex"],
#                     age=row["Age"],
#                     visit=row["Visit"],
#                     modality=row["Modality"],
#                     description=row["Description"],
#                     acquisition_date=row["Acq Date"],
#                     file_format=row["Format"],
#                 )
#                 subjects.append(subject)

#         with open("subjects_list.pkl", "wb") as f:
#             pickle.dump(subjects, f)

#     print(f"Total ADNI subjects: {len(subjects)}")

#     def split_dataset(subjects, shuffle=True):
#         """
#         Split the dataset into training, validation, and test sets.

#         Returns:
#         - train_subjects: list of subjects for training
#         - val_subjects: list of subjects for validation
#         - test_subjects: list of subjects for testing
#         """
#         total_len = len(subjects)
#         train_len = int(0.8 * total_len)
#         val_len = int(0.05 * total_len)
#         test_len = total_len - train_len - val_len

#         if shuffle:
#             subjects = subjects[:]
#             torch.manual_seed(42)  # For reproducibility
#             subjects = random_split(subjects, [train_len, val_len, test_len])[0].dataset

#         train_subjects, val_subjects, test_subjects = random_split(
#             subjects, [train_len, val_len, test_len]
#         )

#         return train_subjects, val_subjects, test_subjects

#     return split_dataset(subjects)


def create_subjects_from_metadata(root_path, metadata_csv_path):
    # Load metadata from CSV
    metadata_df = pd.read_csv(metadata_csv_path)

    # Dictionary to hold torchio.Subject objects per subject_id
    subject_dict = defaultdict(list)

    # First, check if subject_dict.pkl already exists
    subject_dict_path = "/simurgh/u/alanqw/data/fangruih/stru/metadata/adni_t1/paths_and_info_flexpath.pkl"
    subject_dict_path = "/afs/cs.stanford.edu/u/alanqw/rssl/subject_dict_splits_separated_by_subjectid.pkl"
    if os.path.exists(subject_dict_path):
        print(f"Found {subject_dict_path}!")
        with open(subject_dict_path, "rb") as f:
            subject_dict = pickle.load(f)
    else:
        # Iterate over the metadata DataFrame
        print(f"Didn't find {subject_dict_path}. Creating subjects...")
        for i, row in metadata_df.iterrows():
            print(f"{i + 1}/{len(metadata_df)}")
            if row["Group"] not in ["EMCI", "CN", "MCI", "LMCI", "AD"]:
                continue
            image_data_id = row["Image Data ID"]
            subject_id = row["Subject"]

            # Construct the expected directory structure based on the given format
            subject_dir = os.path.join(
                root_path, subject_id, "*", "*", image_data_id, "*.npy"
            )
            glob_paths = glob(subject_dir)

            # Walk through the directory structure to find the .npy file
            for path in glob_paths:
                # Create torchio.Subject with metadata
                subject = tio.Subject(
                    img=tio.ScalarImage(path, reader=numpy_reader),
                    image_data_id=image_data_id,
                    subject=subject_id,
                    group=row["Group"],
                    sex=row["Sex"],
                    age=row["Age"],
                    visit=row["Visit"],
                    modality=row["Modality"],
                    description=row["Description"],
                    acquisition_date=row["Acq Date"],
                    file_format=row["Format"],
                )
                subject_dict[subject_id].append(subject)

        with open(subject_dict_path, "wb") as f:
            pickle.dump(subject_dict, f)

    print(f"Total unique subjects: {len(subject_dict)}")

    with open(subject_dict_path, "rb") as f:
        subject_dict = pickle.load(f)

    # List to hold all group labels
    groups = []

    # Iterate over all subjects and their associated images
    for subject_list in subject_dict.values():
        for subject in subject_list:
            # Access the 'group' attribute of each torchio.Subject
            group = subject["group"]
            groups.append(group)

    # Compute the frequency of each group label
    from collections import Counter

    group_counts = Counter(groups)

    # Display the frequencies
    print("Group Frequencies Across All Images:")
    for group_label, count in group_counts.items():
        print(f"{group_label}: {count}")

    # Now split the subjects into train, val, test
    subject_ids = list(subject_dict.keys())
    random.shuffle(subject_ids)

    total_subjects = len(subject_ids)
    n_train = int(0.80 * total_subjects)
    n_val = int(0.05 * total_subjects)
    n_test = total_subjects - n_train - n_val  # Ensure the sum equals total_subjects

    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train : n_train + n_val]
    test_ids = subject_ids[n_train + n_val :]

    train_subjects = []
    val_subjects = []
    test_subjects = []

    for sid in train_ids:
        train_subjects.extend(subject_dict[sid])

    for sid in val_ids:
        val_subjects.extend(subject_dict[sid])

    for sid in test_ids:
        test_subjects.extend(subject_dict[sid])

    print(f"Total train images: {len(train_subjects)}")
    print(f"Total validation images: {len(val_subjects)}")
    print(f"Total test images: {len(test_subjects)}")

    return train_subjects, val_subjects, test_subjects


def numpy_reader(path):
    data = np.load(path).astype(np.float32)
    affine = np.eye(4)
    return data, affine


def npz_reader(path):
    data = np.load(path)["vol_data"].astype(np.float32)
    affine = np.eye(4)
    return data, affine


class ADNIDataset:
    def __init__(self, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        # root_path = "/scratch/groups/eadeli/data/stru/t1/adni"
        # metadata_csv_path = "/scratch/groups/eadeli/data/stru/t1/metadata/adni/preprocessed_ADNI_mri_2_14_2024.csv"
        root_path = "/simurgh/u/alanqw/data/from_sherlock/scratch/groups/eadeli/data/stru/t1/adni"
        metadata_csv_path = "/simurgh/u/alanqw/data/from_sherlock/scratch/groups/eadeli/data/stru/t1/metadata/adni/preprocessed_ADNI_mri_2_14_2024.csv"
        self.train_subjects, self.val_subjects, self.test_subjects = (
            create_subjects_from_metadata(root_path, metadata_csv_path)
        )

        # Preprocess the metadata
        print("Preprocessing...")
        for subject in self.train_subjects + self.val_subjects + self.test_subjects:
            subject["group"] = adnigroup2int(subject["group"])
            subject["sex"] = adnisex2int(subject["sex"])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seg_available = False

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


class ADNISynthSegDataset:
    def __init__(self, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        root_path = "/simurgh/u/alanqw/data/adni_npy_to_npz"
        self.train_subjects, self.val_subjects, self.test_subjects = (
            self.create_subjects_splits(root_path)
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seg_available = False

    def create_subjects_splits(self, data_dir):
        """
        Creates train, validation, and test datasets from image files and their corresponding segmentations.

        Args:
            data_dir (str): Path to the directory containing image and segmentation files.

        Returns:
            tuple: Three lists containing torchio.Subject instances for training, validation, and testing.
        """
        # Get all files in the directory
        all_files = os.listdir(data_dir)

        # Filter out image files (those not starting with 'synthseg_' and ending with '.nii.gz')
        image_files = [
            f for f in all_files if not f.startswith("synthseg_") and f.endswith(".npz")
        ]

        subjects = []
        for image_file in image_files:
            image_path = os.path.join(data_dir, image_file)
            segmentation_filename = "synthseg_" + image_file
            segmentation_path = os.path.join(data_dir, segmentation_filename)

            if os.path.exists(segmentation_path):
                subject = tio.Subject(
                    img=tio.ScalarImage(image_path, reader=npz_reader),
                    seg=tio.LabelMap(segmentation_path, reader=npz_reader),
                )
                subjects.append(subject)

        # Shuffle the subjects
        random.shuffle(subjects)

        # Calculate split sizes
        total_subjects = len(subjects)
        train_size = int(0.80 * total_subjects)
        val_size = int(0.05 * total_subjects)

        # Split the subjects
        train_subjects = subjects[:train_size]
        val_subjects = subjects[train_size : train_size + val_size]
        test_subjects = subjects[train_size + val_size :]

        return train_subjects, val_subjects, test_subjects

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
