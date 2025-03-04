import os
import torchio as tio
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import itertools
from .utils import PairedDataset
from glob import glob
import pickle
from collections import defaultdict
import random

def ppmigroup2int(group):
    mapping = {"Control": 0, "PD": 1}
    return torch.tensor(mapping[group]).long()

def ppmisex2int(sex):
    # mapping = {"M": 0, "F": 1}
    mapping = {0.0 : 0, 1.0 : 1} # Used with get_pkl_data
    return torch.tensor(mapping[sex]).long()

def create_subjects_from_metadata(root_path, metadata_csv_path):
    # Load metadata from CSV
    metadata_df = pd.read_csv(metadata_csv_path)
    
    # Dictionary to hold torchio.Subject objects per subject_id
    subject_dict = defaultdict(list)
    
    # First, check if subject_dict.pkl already exists
    subject_dict_path = "/simurgh/u/alanqw/data/fangruih/stru/metadata/ppmi_t1/paths_and_info_flexpath.pkl"
    if os.path.exists(subject_dict_path):
        print(f"Found {subject_dict_path}!")
        with open(subject_dict_path, "rb") as f:
            subject_dict = pickle.load(f)
    else:
        # Iterate over the metadata DataFrame
        print(f"Didn't find {subject_dict_path}. Creating subjects...")
        for i, row in metadata_df.iterrows():
            print(f"{i + 1}/{len(metadata_df)}")
            if row["Group"] not in ["Control", "PD"]:
                continue
            image_data_id = row["Image Data ID"]
            subject_id = str(row["Subject"])
            
            # Construct the expected directory structure based on the given format
            subject_dir = os.path.join(
                root_path, subject_id, "*", "*", image_data_id, "*.npy"
            )
            glob_paths = glob(subject_dir)
            if not glob_paths:
                print(f"No files found for subject {subject_id} with Image Data ID {image_data_id}")
                continue

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

def get_pkl_data(file_path="/scratch/groups/eadeli/data/stru_new/t1/metadata/ppmi/paths_and_info.pkl", file_dir_prefix="/scratch/groups/eadeli/data/SC/stru/t1/other_datasets/ppmi"):
    # Load the data from the specified .pkl file
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        print(f"Processing dataset: {file_path}")
        print(f"Available splits: {data.keys()}")

        # Initialize lists to hold the subjects
        train_subjects = []
        val_subjects = []
        test_subjects = []

        # Process each data split
        for split in ["train", "val", "test"]:
            if split in data:
                print(f"Processing {split} split")

                # Extract paths, ages, and sexes
                paths = data[split]["paths"]
                ages = data[split]["ages"]
                sexes = data[split]["sexes"]
                groups = data[split]["groups"]

                # Convert to lists if they are NumPy arrays
                paths = paths.tolist() if isinstance(paths, np.ndarray) else paths
                ages = ages.tolist() if isinstance(ages, np.ndarray) else ages
                sexes = sexes.tolist() if isinstance(sexes, np.ndarray) else sexes
                groups = groups.tolist() if isinstance(groups, np.ndarray) else groups

                # Adjust the file paths
                paths = [os.path.join(file_dir_prefix, path) for path in paths]

                # Create subjects
                subjects = []
                for path, age, sex, group in zip(paths, ages, sexes, groups):
                    if group not in ["PD", "Control"]:
                        continue
                    subject = tio.Subject(
                        img=tio.ScalarImage(path, reader=numpy_reader),
                        age=age,
                        sex=sex,
                        group=group,
                    )
                    subjects.append(subject)

                # Add subjects to the corresponding split list
                if split == "train":
                    train_subjects.extend(subjects)
                elif split == "val":
                    val_subjects.extend(subjects)
                elif split == "test":
                    test_subjects.extend(subjects)

        # Iterate over all subjects and their associated images
        all_groups = []
        for subject in train_subjects + val_subjects + test_subjects:
            # Access the 'group' attribute of each torchio.Subject
            group = subject["group"]
            all_groups.append(group)

        # Compute the frequency of each group label
        from collections import Counter

        group_counts = Counter(all_groups)
        # Debug output to check the results
        print("Group Frequencies Across All Images:")
        for group_label, count in group_counts.items():
            print(f"{group_label}: {count}")

        print(f"Total train subjects: {len(train_subjects)}")
        print(f"Total val subjects: {len(val_subjects)}")
        print(f"Total test subjects: {len(test_subjects)}")

    return train_subjects, val_subjects, test_subjects

def numpy_reader(path):
    data = np.load(path).astype(np.float32)
    affine = np.eye(4)
    return data, affine

class PPMIDataset:
    def __init__(self, data_path, batch_size=8, num_workers=4, shuffle=True):
        """
        Initialize the PPMI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        root_path = data_path
        metadata_csv_path = "/scratch/groups/eadeli/data/SC/stru/t1/other_datasets/metadata/ppmi_t1/ppmi_mri_preprocessed_2_14_2024.csv"
        self.train_subjects, self.val_subjects, self.test_subjects = (
            # create_subjects_from_metadata(root_path, metadata_csv_path)
            get_pkl_data(file_dir_prefix=root_path)
        )

        # Preprocess the metadata
        print("Preprocessing...")
        for subject in self.train_subjects + self.val_subjects + self.test_subjects:
            subject["group"] = F.one_hot(ppmigroup2int(subject["group"]), num_classes=2)
            subject["sex"] = F.one_hot(ppmisex2int(subject["sex"]), num_classes=2)
            subject["age"] = torch.tensor(subject["age"])[..., None]
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

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
