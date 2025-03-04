import os
import torchio as tio
from torch.utils.data import DataLoader, Dataset
import random
import re
import itertools
from collections import defaultdict
from dataset.utils import ConcatDataset, PairedDataset


class NNUnetDataset:
    def __init__(
        self,
        root_dir,
        include_seg=True,
        include_lesion_seg=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.include_seg = include_seg
        self.include_lesion_seg = include_lesion_seg
        self.must_have_longitudinal = False
        self.seg_available = True

    def get_subjects(
        self,
        train: bool,
    ):
        """Creates dictionary of TorchIO subjects.
        {
            'sub1': {
                'mod1': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
                'mod2': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
                ...
            },
            'sub2': {
                'mod1': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
                'mod2': [/path/to/subject-time1.nii.gz, /path/to/subject-time2.nii.gz, ...],
                ...
            },
            ...
        }
        Keys are each unique subject and modality, values are list of all paths with that subject and modality.
        Should be 3 or 4 different timepoints.

        If dataset has timepoints, pathnames are of the form: "<dataset>-time<time_id>_<sub_id>_<mod_id>.nii.gz."
        If dataset has no timepoints, pathnames are of the form: "<dataset>_<sub_id>_<mod_id>.nii.gz."

        TODO: Currently, code assumes all subjects have SynthSeg-generated labels.
        """
        if train:
            split_folder_img, split_folder_seg = "imagesTr", "synthSeglabelsTr"
        else:
            split_folder_img, split_folder_seg = "imagesTs", "synthSeglabelsTs"

        img_data_folder = os.path.join(self.root_dir, split_folder_img)
        seg_data_folder = os.path.join(self.root_dir, split_folder_seg)
        if self.include_lesion_seg:
            split_folder_lesion_seg = "labelsTr"
            lesion_seg_data_folder = os.path.join(
                self.root_dir, split_folder_lesion_seg
            )
            assert os.path.exists(
                lesion_seg_data_folder
            ), "No lesion segmentation found in labelsTr folder."

        # Initialize an empty dictionary using defaultdict for easier nested dictionary handling
        data_structure = defaultdict(lambda: defaultdict(list))

        # Regex pattern to match the filename and extract time_id, subject_id, and modality_id
        # Makes the time part optional and defaults to 0 if not present
        pattern = re.compile(r"(?:-time(\d+))?_(\d+)_(\d+)\.nii\.gz")

        def add_to_dict(subject_id, modality, subject):
            # Append the file path to the list, will sort later
            data_structure[subject_id][modality].append(subject)

        # Iterate through each file in the directory
        for filename in os.listdir(img_data_folder):
            if "mask" in filename or not filename.endswith(".nii.gz"):
                continue
            img_path = os.path.join(img_data_folder, filename)
            seg_path = os.path.join(seg_data_folder, filename)
            # Only include subjects that have a corresponding segmentation file
            if not os.path.exists(seg_path):
                continue
            if self.include_seg:
                # Construct TorchIO subject
                subject_kwargs = {
                    "img": tio.ScalarImage(img_path),
                    "seg": tio.LabelMap(seg_path),
                }
            else:
                subject_kwargs = {
                    "img": tio.ScalarImage(img_path),
                }
            if self.include_lesion_seg:
                lesion_seg_path = os.path.join(lesion_seg_data_folder, filename)
                subject_kwargs["lesion_seg"] = tio.LabelMap(lesion_seg_path)
            subject = tio.Subject(**subject_kwargs)

            match = pattern.search(filename)
            if match:
                # Extract time_id, subject_id, and modality_id based on regex groups
                # Default time_id to '0' if not present
                time_id, subject_id, modality_id = match.groups(default="0")
                # Add the extracted information along with the file path to the dictionary
                add_to_dict(subject_id, modality_id, subject)
            subject["modality"] = modality_id

        if self.must_have_longitudinal:
            for subject in list(data_structure.keys()):
                for modality in list(data_structure[subject].keys()):
                    # Check if there are at least 2 different timepoints for the modality
                    if len(data_structure[subject][modality]) < 2:
                        del data_structure[subject][
                            modality
                        ]  # Remove modality if it doesn't meet the criterion
                if not data_structure[
                    subject
                ]:  # Check if the subject has no remaining modalities
                    del data_structure[
                        subject
                    ]  # Remove subject if it has no valid modalities
        if len(data_structure) == 0 and self.must_have_longitudinal:
            raise ValueError(
                f"No subjects with longitudinal data found in {self.root_dir}. Do you have time- information in your paths?"
            )
        if len(data_structure) == 0:
            raise ValueError(f"No subjects found in {self.root_dir}.")

        # Convert defaultdicts back to regular dicts for cleaner output or further processing
        subject_dict = {k: dict(v) for k, v in data_structure.items()}

        # Also create a flattened list of all paths for convenience
        subject_list = []
        for subject in subject_dict:
            for modality in subject_dict[subject]:
                subject_list.extend(subject_dict[subject][modality])
        return subject_dict  # , subject_list

    def get_reference_subject(self, transform, train=True):
        """Get a reference subject with all modalities"""
        subjects = self.get_subjects(train)
        ref_paths = subjects["000035"]
        # Build tio.Subject
        subject_dict = {}
        for mod in ref_paths:
            subject_dict[mod] = transform(
                ref_paths[mod][0]
            )  # [0] because ref_paths[mod] is a list of longitudinal paths
        return subject_dict

    def aggregate_paths_by_modality(self, data_structure):
        modality_aggregate = {}
        for subject, modalities in data_structure.items():
            for modality, paths in modalities.items():
                if modality not in modality_aggregate:
                    modality_aggregate[modality] = []
                modality_aggregate[modality].extend(paths)

        # Ensure that we only consider modalities with at least two paths
        valid_modalities = {
            mod: paths for mod, paths in modality_aggregate.items() if len(paths) >= 2
        }
        return valid_modalities

    def get_pretrain_loader(self, batch_size, num_workers, transform, paired=False):
        subjects = self.get_subjects(
            train=True,
        )
        modality_dict = self.aggregate_paths_by_modality(subjects)

        pretrain_datasets = []
        for subjects_list in modality_dict.values():
            if paired:
                pairs_list = list(itertools.product(subjects_list, subjects_list))
                pretrain_datasets.append(
                    PairedDataset(
                        pairs_list,
                        transform=transform,
                    )
                )

            else:
                pretrain_datasets.append(
                    tio.data.SubjectsDataset(
                        subjects_list,
                        transform=transform,
                    )
                )
        print(pretrain_datasets)
        pretrain_dataset = ConcatDataset(pretrain_datasets)

        return DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def get_train_loader(self, batch_size, num_workers, transform, paired=False):
        subjects = self.get_subjects(
            train=True,
        )
        if isinstance(subjects, dict):
            # pretrain_datasets = [
            #     tio.data.SubjectsDataset(
            #         subjects_list,
            #         transform=transform,
            #     )
            #     for subjects_list in subjects.values()
            # ]
            # pretrain_dataset = ConcatDataset(pretrain_datasets)
            # Create a list to hold all the Subject objects
            subject_dict = subjects
            all_subjects = []

            # Iterate over the dictionary
            for subject_id, modalities in subjects.items():
                for modality_id, subjects in modalities.items():
                    all_subjects.extend(subjects)
            pretrain_dataset = tio.data.SubjectsDataset(
                all_subjects, transform=transform
            )
        else:
            pretrain_dataset = tio.data.SubjectsDataset(
                subjects[0] + subjects[1],
                transform=transform,
            )

        return DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def get_val_loader(self, batch_size, num_workers, transform, paired=False):
        subjects = self.get_subjects(
            train=False,
        )
        if isinstance(subjects, dict):
            # pretrain_datasets = [
            #     tio.data.SubjectsDataset(
            #         subjects_list,
            #         transform=transform,
            #     )
            #     for subjects_list in subjects.values()
            # ]
            # pretrain_dataset = ConcatDataset(pretrain_datasets)
            # Create a list to hold all the Subject objects
            subject_dict = subjects
            all_subjects = []

            # Iterate over the dictionary
            for subject_id, modalities in subjects.items():
                for modality_id, subjects in modalities.items():
                    all_subjects.extend(subjects)
            pretrain_dataset = tio.data.SubjectsDataset(
                all_subjects, transform=transform
            )
        else:
            pretrain_dataset = tio.data.SubjectsDataset(
                subjects[0] + subjects[1],
                transform=transform,
            )

        return DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def get_test_loaders(
        self, batch_size, num_workers, transform, list_of_mods, paired=False
    ):
        subjects = self.get_subjects(
            train=False,
        )

        if isinstance(subjects, dict):
            # Convert subject dict to modality dict
            modality_dict = {}
            for subject_id, modalities in subjects.items():
                for modality_id, subjects in modalities.items():
                    if modality_id not in modality_dict:
                        modality_dict[modality_id] = []
                    modality_dict[modality_id].extend(subjects)

            test_datasets = []
            for dataset_name in list_of_mods:
                mod1, mod2 = self._parse_test_mod(dataset_name)
                subjects1 = modality_dict[mod1]
                subjects2 = modality_dict[mod2]
                test_datasets.append(
                    PairedDataset(list(zip(subjects1, subjects2)), transform=transform)
                )
            test_dataset = ConcatDataset(test_datasets)

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            test_dataset = PairedDataset(
                list(zip(subjects[0], subjects[1])), transform=transform
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return test_loader

    def get_loaders(
        self, batch_size, num_workers, mix_modalities, transform, list_of_test_mods
    ):
        return (
            self.get_pretrain_loader(batch_size, num_workers, transform),
            self.get_train_loader(batch_size, num_workers, transform),
            self.get_test_loaders(
                batch_size, num_workers, transform, list_of_test_mods
            ),
        )
