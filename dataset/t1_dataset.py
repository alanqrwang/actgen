import pickle
import numpy as np
import os
import torchio as tio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
from .utils import PairedDataset
import pickle


def numpy_reader(path):
    data = np.load(path).astype(np.float32)
    affine = np.eye(4)
    return data, affine


def get_subject_dicts(
    # file_dir_prefix="/scr/fangruih/stru",
    file_dir_prefix="/simurgh/u/alanqw/data/fangruih/stru",
):
    # Define the data directory prefix and dataset names
    prefix = "/simurgh/u/alanqw/data"
    dataset_names = [
        f"{prefix}/metadata/abcd/paths_and_info_flexpath.pkl",
        f"{prefix}/metadata/adni_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/metadata/hcp_ag_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/metadata/hcp_dev_t1/paths_and_info_flexpath.pkl",
        f"{prefix}/metadata/hcp_ya_mpr1/paths_and_info_flexpath.pkl",
        f"{prefix}/metadata/ppmi_t1/paths_and_info_flexpath.pkl",
    ]

    # Initialize lists to hold the subjects
    train_subjects = []
    val_subjects = []
    test_subjects = []

    for dataset_name in dataset_names:
        with open(dataset_name, "rb") as file:
            data = pickle.load(file)
            print(f"Processing dataset: {dataset_name}")
            print(f"Available splits: {data.keys()}")

            for split in ["train", "val", "test"]:
                if split in data:
                    print(f"Processing {split} split")
                    # Extract paths, ages, and sexes
                    paths = data[split]["paths"]
                    ages = data[split]["age"]
                    sexes = data[split]["sex"]

                    # Convert to lists if they are NumPy arrays
                    paths = paths.tolist() if isinstance(paths, np.ndarray) else paths
                    ages = ages.tolist() if isinstance(ages, np.ndarray) else ages
                    sexes = sexes.tolist() if isinstance(sexes, np.ndarray) else sexes

                    # Adjust the file paths
                    paths = [os.path.join(file_dir_prefix, path) for path in paths]

                    # Create subjects
                    subjects = []
                    for path, age, sex in zip(paths, ages, sexes):
                        subject = tio.Subject(
                            img=tio.ScalarImage(path, reader=numpy_reader),
                            age=age,
                            sex=sex,
                        )
                        subjects.append(subject)

                    # Add subjects to the corresponding split list
                    if split == "train":
                        train_subjects.extend(subjects)
                    elif split == "val":
                        val_subjects.extend(subjects)
                    elif split == "test":
                        test_subjects.extend(subjects)

            # Debug output to check the results
            print(f"Total train subjects so far: {len(train_subjects)}")
            print(f"Total val subjects so far: {len(val_subjects)}")
            print(f"Total test subjects so far: {len(test_subjects)}")

    # Optionally, compute age normalization parameters using train subjects
    # Uncomment the following lines if you want to normalize the ages
    # train_ages = np.array([subject.age for subject in train_subjects])
    # age_mean = np.mean(train_ages)
    # age_std = np.std(train_ages)
    # # Normalize ages for all subjects
    # for subject in train_subjects + val_subjects + test_subjects:
    #     subject.age = (subject.age - age_mean) / age_std

    return train_subjects, val_subjects, test_subjects


class T1Dataset:
    def __init__(self):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        self.train_subjects, self.val_subjects, self.test_subjects = get_subject_dicts()

        # Preprocess the metadata
        print("Preprocessing...")
        for subject in self.train_subjects + self.val_subjects + self.test_subjects:
            subject["age"] = torch.tensor(subject["age"])
            subject["sex"] = F.one_hot(torch.tensor(subject["sex"]).long(), 2)
        self.seg_available = False

    def get_reference_subject(self, transform=None):
        return {"0000": self.train_subjects[0]}

    def get_pretrain_loader(
        self, batch_size, num_workers, transform=None, paired=False
    ):
        return self.get_train_loader(
            batch_size, num_workers, transform=transform, paired=paired
        )

    def _get_loader(
        self,
        subjects,
        batch_size,
        num_workers,
        shuffle=False,
        transform=None,
        paired=False,
    ):
        if paired:
            paired_subjects = list(itertools.product(subjects, subjects))
            dataset = PairedDataset(paired_subjects, transform=transform)
        else:
            dataset = tio.SubjectsDataset(subjects, transform=transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def get_train_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the training set.

        Returns:
        - train_loader: DataLoader for the training set
        """
        return self._get_loader(
            self.train_subjects,
            batch_size,
            num_workers,
            shuffle=True,
            transform=transform,
            paired=paired,
        )

    def get_val_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the validation set.

        Returns:
        - val_loader: DataLoader for the validation set
        """
        return self._get_loader(
            self.val_subjects,
            batch_size,
            num_workers,
            shuffle=False,
            transform=transform,
            paired=paired,
        )

    def get_test_loader(self, batch_size, num_workers, transform=None, paired=False):
        """
        Get the DataLoader for the test set.

        Returns:
        - test_loader: DataLoader for the test set
        """
        return self._get_loader(
            self.test_subjects,
            batch_size,
            num_workers,
            shuffle=False,
            transform=transform,
            paired=paired,
        )
