import os
from torch.utils.data import DataLoader
import pandas as pd
from collections import defaultdict
import torchio as tio
import re
from torch.utils.data import Dataset
import random

id_csv_file = "/afs/cs.stanford.edu/u/alanqw/rssl/dataset/gigamed_id.csv"
ood_csv_file = "/afs/cs.stanford.edu/u/alanqw/rssl/dataset/gigamed_ood.csv"


def read_subjects_from_disk(
    root_dir: str,
    train: bool,
    include_seg: bool = True,
    include_lesion_seg: bool = False,
    must_have_longitudinal=False,
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

    img_data_folder = os.path.join(root_dir, split_folder_img)
    seg_data_folder = os.path.join(root_dir, split_folder_seg)
    if include_lesion_seg:
        split_folder_lesion_seg = "labelsTr"
        lesion_seg_data_folder = os.path.join(root_dir, split_folder_lesion_seg)
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
        if include_seg:
            # Construct TorchIO subject
            subject_kwargs = {
                "img": tio.ScalarImage(img_path),
                "seg": tio.LabelMap(seg_path),
            }
        else:
            subject_kwargs = {
                "img": tio.ScalarImage(img_path),
            }
        if include_lesion_seg:
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

    if must_have_longitudinal:
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
    if len(data_structure) == 0 and must_have_longitudinal:
        raise ValueError(
            f"No subjects with longitudinal data found in {root_dir}. Do you have time- information in your paths?"
        )
    if len(data_structure) == 0:
        raise ValueError(f"No subjects found in {root_dir}.")

    # Convert defaultdicts back to regular dicts for cleaner output or further processing
    subject_dict = {k: dict(v) for k, v in data_structure.items()}

    # Also create a flattened list of all paths for convenience
    subject_list = []
    for subject in subject_dict:
        for modality in subject_dict[subject]:
            subject_list.extend(subject_dict[subject][modality])
    return subject_dict, subject_list


class SingleSubjectDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train,
        transform=None,
        include_seg=True,
        include_lesion_seg=False,
    ):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir,
            train,
            include_seg=include_seg,
            include_lesion_seg=include_lesion_seg,
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        sub = random.choice(self.subject_list)
        sub.load()
        if self.transform:
            sub = self.transform(sub)
        return sub


class PairedDataset(Dataset):
    """General paired dataset.
    Given subject list, samples pairs of subjects without restriction."""

    def __init__(
        self,
        root_dir,
        train,
        transform=None,
        include_seg=True,
        include_lesion_seg=False,
    ):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir,
            train,
            include_seg=include_seg,
            include_lesion_seg=include_lesion_seg,
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        sub1 = random.sample(self.subject_list, 1)[0]
        sub2 = random.sample(self.subject_list, 1)[0]
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class SSSMPairedDataset(Dataset):
    """Longitudinal paired dataset.
    Given subject list, samples same-subject, single-modality pairs."""

    def __init__(self, root_dir, train, transform=None, include_seg=True):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg, must_have_longitudinal=True
        )
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        # Randomly select a subject
        subject = random.choice(list(self.subject_dict.keys()))
        # Randomly select a modality for the chosen subject
        modality = random.choice(list(self.subject_dict[subject].keys()))
        # Randomly sample two paths from the chosen subject and modality
        sub1, sub2 = random.sample(self.subject_dict[subject][modality], 2)
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class DSSMPairedDataset(Dataset):
    """DSSM paired dataset.
    Given subject list, samples different-subject, same-modality pairs."""

    def __init__(self, root_dir, train, transform=None, include_seg=True):
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg
        )

        def aggregate_paths_by_modality(data_structure):
            modality_aggregate = {}
            for subject, modalities in data_structure.items():
                for modality, paths in modalities.items():
                    if modality not in modality_aggregate:
                        modality_aggregate[modality] = []
                    modality_aggregate[modality].extend(paths)

            # Ensure that we only consider modalities with at least two paths
            valid_modalities = {
                mod: paths
                for mod, paths in modality_aggregate.items()
                if len(paths) >= 2
            }
            return valid_modalities

        self.modality_dict = aggregate_paths_by_modality(self.subject_dict)
        # assert (
        #     len(self.modality_dict) > 1
        # ), f"Must have at least 2 modalities: {root_dir}"
        self.transform = transform

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        mult_mod_list = list(self.modality_dict.values())
        single_mod_list = random.sample(mult_mod_list, 1)[0]
        sub1, sub2 = random.sample(single_mod_list, 2)
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


class LongitudinalPathDataset:
    """At every iteration, returns all paths associated with the same subject and same modality.
    Relies on TorchIO's lazy loading. If no transform is performed, then TorchIO
    won't load the image data into memory."""

    def __init__(self, root_dir, train, transform=None, include_seg=True, group_size=4):
        # super().__init__(root_dir, train, include_seg)
        self.subject_dict, self.subject_list = read_subjects_from_disk(
            root_dir, train, include_seg=include_seg, must_have_longitudinal=True
        )
        self.transform = transform  # This is hardcoded to None
        self.group_size = group_size

    def get_total_subjects(self):
        return len(self.subject_dict)

    def get_total_images(self):
        return len(self.subject_list)

    def __len__(self):
        return len(self.subject_dict)

    def __getitem__(self, x):
        # Randomly select a subject
        subject = random.choice(list(self.subject_dict.keys()))
        # Randomly select a modality for the chosen subject
        modality = random.choice(list(self.subject_dict[subject].keys()))
        # Randomly sample paths from the chosen subject and modality
        single_sub_mod_list = self.subject_dict[subject][modality]
        subs = random.sample(
            single_sub_mod_list, min(len(single_sub_mod_list), self.group_size)
        )
        from torchio.data import SubjectsDataset as TioSubjectsDataset

        return SimpleDatasetIterator(TioSubjectsDataset(subs, transform=self.transform))


class AggregatedFamilyDataset(Dataset):
    """Aggregates multiple ``families'' of datasets into one giant dataset.
    Also appends the name of the family to each sample.

    A family is defined as a list of datasets which share some characteristic.
    """

    def __init__(self, list_of_dataset_families, names):
        """Samples uniformly over multiple families.
        Then, samples uniformly within that family.

        Inputs:
            list_of_dataset_families: A list of lists of datasets.
            names: A list of names for each list of datasets.
        """
        self.list_of_dataset_families = list_of_dataset_families
        self.names = names
        assert len(list_of_dataset_families) == len(names)
        self.num_families = len(list_of_dataset_families)

    def __getitem__(self, i):
        family_idx = random.randrange(self.num_families)
        family = self.list_of_dataset_families[family_idx]
        family_name = self.names[family_idx]
        dataset = random.choice(family)
        return dataset[i]  # , family_name

    def __len__(self):
        l = 0
        for d in self.list_of_dataset_families:
            for sub_ds in d:
                l += len(sub_ds)
        return l


class SimpleDatasetIterator:
    """Simple replacement to DataLoader"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # Reset the index each time iter is called.
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            item = self.dataset[self.index]
            self.index += 1
            return item
        else:
            # No more data, stop the iteration.
            raise StopIteration


class GigaMedPaths:
    """Convenience class. Handles all dataset names in GigaMed."""

    def __init__(self):
        self.gigamed_id_df = pd.read_csv(id_csv_file, header=0)
        self.gigamed_ood_df = pd.read_csv(ood_csv_file, header=0)

    @staticmethod
    def get_filtered_ds_paths(df, conditions):
        """
        Returns a list of ds_path values that satisfy the given conditions.

        Parameters:
        - file_path: The path to the CSV file.
        - conditions: A dictionary where keys are column names and values are the conditions those columns must satisfy.

        Returns:
        - A list of ds_path values that meet the conditions.
        """
        # Apply each condition in the conditions dictionary
        for column, value in conditions.items():
            df = df[df[column] == value]
        # Extract the list of ds_path that satisfy the conditions
        return df["ds_path"].tolist()

    def get_ds_dirs(self, conditions, id=True):
        df = self.gigamed_id_df if id else self.gigamed_ood_df
        return self.get_filtered_ds_paths(df, conditions)


class GigaMedDataset:
    """Convenience class. Handles creating Pytorch Datasets."""

    def __init__(
        self,
        include_seg=True,
        transform=None,
    ):
        self.include_seg = include_seg
        self.transform = transform
        self.gigamed_paths = GigaMedPaths()

    def get_dataset_family(self, conditions, DatasetType, id=True, **dataset_kwargs):
        ds_dirs = self.gigamed_paths.get_ds_dirs(conditions, id=id)
        train = conditions.get("has_train", False)
        datasets = {}
        for ds_dir in ds_dirs:
            datasets[ds_dir] = DatasetType(
                ds_dir,
                train,
                self.transform,
                include_seg=self.include_seg,
                **dataset_kwargs,
            )
        return datasets

    def get_reference_subject(self):
        conditions = {}
        ds_name = self.gigamed_paths.get_ds_dirs(conditions, id=True)[0]
        root_dir = os.path.join(ds_name)
        _, subject_list = read_subjects_from_disk(root_dir, True, include_seg=False)
        return subject_list[0]


class GigaMed:
    """Top-level class. Handles creating Pytorch dataloaders.

    Reads data from:
      1) /midtier/sablab/scratch/alw4013/data/nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/
      2) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_preprocessed/
      3) /midtier/sablab/scratch/alw4013/data/brain_nolesions_nnUNet_1mmiso_256x256x256_MNI_HD-BET_preprocessed/

    Data from all directories are nnUNet-like in structure.
    Data from 2) are not skull-stripped. Data from 1) and 3) are skull-stripped using HD-BET.
    Data from 1) have (extreme) lesions. Data from 3) do not have lesions.

    Our data is split along 2 dimensions: skullstripped vs. non-skullstripped, and normal brains vs. brains with (extreme) lesions.
    ---------------------------------------------------------
    | Skullstripped, normal     | Skullstripped, lesion     |
    | Dice Loss or MSE Loss     | MSE Loss                  |
    | Any transform             | Rigid or Affine           |
    ---------------------------------------------------------
    | Non-skullstripped, normal | Non-skullstripped, lesion |
    | Dice Loss                 | -----                     |
    | Any transform             | -----                     |
    ---------------------------------------------------------

    Rules:
        1) If Dice loss, sample pairs without restriction.
        2) If MSE loss, sample same-modality pairs.
            a) If longitudinal pairs, use rigid transformation.
            b) If cross-subject with lesions, use affine transformation.
            c) If cross-subject with normal, use TPS_logunif.

    If data is skullstripped and normal, then we:
        1) Loss: can generate segmentations using SynthSeg and minimize Dice loss
            (we can use MSE loss on images directly, but won't for simplicity. Also Dice loss is more robust to noise).
        2) Transformation: can use a more flexible transformation (i.e. TPS) during training.

    If data is not skullstripped and normal, then we:
        1) Loss: cannot use MSE loss and must use Dice loss, because skull and neck are highly variable.
        2) Transformation: can use a more flexible transformation (i.e. TPS) during training.

    If data is skullstripped and has lesions, then we:
        1) Loss: cannot rely on the quality of SynthSeg labels and must use MSE loss.
        2) Transformation: must use a restrictive transformation (i.e. rigid and affine) during training.

    If data is not skullstripped and has lesions, then we:
        1) Loss: cannot use MSE loss or Dice loss


    In summary, there are 3 settings at training:
      1) Skullstripped and normal: Dice loss or MSE loss, TPS_logunif
      2) Skullstripped and lesion: MSE loss, rigid or affine
      3) Non-skullstripped and normal: Dice loss, TPS_logunif
    """

    def __init__(
        self,
        batch_size,
        num_workers,
        include_seg=True,
        transform=None,
        same_mod_training=False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.same_mod_training = same_mod_training
        self.seg_available = include_seg

        self.dataset = GigaMedDataset(include_seg=include_seg, transform=transform)

    def print_dataset_stats(self, datasets, prefix=""):
        print(f"\n\n{prefix} dataset family has {len(datasets)} datasets.")
        # print("Conditions:")
        # pprint(conditions)
        # print(str(DatasetType))
        tot_sub = 0
        tot_img = 0
        for name, ds in datasets.items():
            tot_sub += len(ds)
            tot_img += ds.get_total_images()
            print(
                f"-> {name} has {len(ds)} subjects and {ds.get_total_images()} images."
            )
        print("Total subjects: ", tot_sub)
        print("Total images: ", tot_img)

    def get_train_loader(self, *args, **kwargs):
        dataset_type = DSSMPairedDataset if self.same_mod_training else PairedDataset
        all_datasets_same_mod = self.dataset.get_dataset_family(
            {
                "has_train": True,
                "is_synthetic": False,
                "is_skullstripped": True,
            },
            dataset_type,
            id=True,
        )
        family_datasets = [
            all_datasets_same_mod,
        ]
        family_names = [
            "diff_sub_same_mod",
        ]

        # Print some stats
        for name, ds in zip(family_names, family_datasets):
            self.print_dataset_stats(ds, f"TRAIN: {name}")

        family_datasets = [list(l.values()) for l in family_datasets]
        final_dataset = AggregatedFamilyDataset(family_datasets, family_names)

        train_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def get_pretrain_loader(self, *args, **kwargs):
        """Pretrain on all datasets."""
        if "paired" in kwargs and kwargs["paired"]:
            return self.get_train_loader(*args, **kwargs)

        else:
            datasets = self.dataset.get_dataset_family(
                {"has_train": True, "is_synthetic": False},
                SingleSubjectDataset,
                id=True,
            )
            self.print_dataset_stats(datasets, "PRETRAIN:")
            family_datasets = [
                list(datasets.values()),
            ]
            family_names = [
                "all_datasets",
            ]

            final_dataset = AggregatedFamilyDataset(family_datasets, family_names)

            pretrain_loader = DataLoader(
                final_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            return pretrain_loader

    def get_val_loader(self, ss=True, id=True, *args, **kwargs):
        datasets = self.dataset.get_dataset_family(
            {
                "has_test": True,
                "is_synthetic": False,
                "has_lesion": False,
                "is_skullstripped": ss,
            },
            SingleSubjectDataset,
            id=id,
        )

        self.print_dataset_stats(datasets, f"EVAL, skullstripped={ss}, Normal")

        family_datasets = [
            list(datasets.values()),
        ]
        family_names = [
            "all_datasets",
        ]

        final_dataset = AggregatedFamilyDataset(family_datasets, family_names)

        val_loader = DataLoader(
            final_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return val_loader

    def get_test_loader(self, ss=True, id=True, *args, **kwargs):
        return self.get_val_loader(ss, id, *args, **kwargs)

    # def get_eval_group_loaders(self, ss, id=True):
    #     datasets = self.dataset.get_dataset_family(
    #         {
    #             "has_test": True,
    #             "is_synthetic": False,
    #             "has_lesion": False,
    #             "is_skullstripped": ss,
    #         },
    #         SingleSubjectDataset,
    #         id=id,
    #     )

    #     self.print_dataset_stats(datasets, f"EVAL, skullstripped={ss}, Normal group")

    #     loaders = {}
    #     for name, ds in datasets.items():
    #         loaders[name] = DataLoader(
    #             ds,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers,
    #         )
    #     return loaders

    # def get_eval_longitudinal_loaders(self, ss, id=True):
    #     datasets = self.dataset.get_dataset_family(
    #         {
    #             "has_test": True,
    #             "is_synthetic": False,
    #             "has_lesion": False,
    #             "is_skullstripped": ss,
    #             "has_longitudinal": True,
    #         },
    #         LongitudinalPathDataset,
    #         id=id,
    #     )

    #     self.print_dataset_stats(
    #         datasets, f"EVAL, skullstripped={ss}, Normal longitudinal"
    #     )

    #     loaders = {}
    #     for name, ds in datasets.items():
    #         loaders[name] = SimpleDatasetIterator(
    #             ds,
    #         )
    #     return loaders

    # def get_eval_lesion_loaders(self, ss, id=True):
    #     datasets = self.dataset.get_dataset_family(
    #         {
    #             "has_test": True,
    #             "is_synthetic": False,
    #             "has_lesion": True,
    #             "is_skullstripped": ss,
    #         },
    #         SingleSubjectDataset,
    #         id=id,
    #     )

    #     self.print_dataset_stats(datasets, f"EVAL, skullstripped={ss}, Lesion")

    #     loaders = {}
    #     for name, ds in datasets.items():
    #         loaders[name] = DataLoader(
    #             ds,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers,
    #         )
    #     return loaders

    def get_reference_subject(self, *args, **kwargs):
        return self.dataset.get_reference_subject()


if __name__ == "__main__":
    datasets_with_longitudinal = [
        "Dataset5114_UCSF-ALPTDG",
        "Dataset6000_PPMI-T1-3T-PreProc",
        "Dataset6001_ADNI-group-T1-3T-PreProc",
        "Dataset6002_OASIS3",
    ]
    datasets_with_multiple_modalities = [
        "Dataset4999_IXIAllModalities",
        "Dataset5000_BraTS-GLI_2023",
        "Dataset5001_BraTS-SSA_2023",
        "Dataset5002_BraTS-MEN_2023",
        "Dataset5003_BraTS-MET_2023",
        "Dataset5004_BraTS-MET-NYU_2023",
        "Dataset5005_BraTS-PED_2023",
        "Dataset5006_BraTS-MET-UCSF_2023",
        "Dataset5007_UCSF-BMSR",
        "Dataset5012_ShiftsBest",
        "Dataset5013_ShiftsLjubljana",
        "Dataset5038_BrainTumour",
        "Dataset5090_ISLES2022",
        "Dataset5095_MSSEG",
        "Dataset5096_MSSEG2",
        "Dataset5111_UCSF-ALPTDG-time1",
        "Dataset5112_UCSF-ALPTDG-time2",
        "Dataset5113_StanfordMETShare",
        "Dataset5114_UCSF-ALPTDG",
    ]
    datasets_with_one_modality = [
        "Dataset5010_ATLASR2",
        "Dataset5041_BRATS",
        "Dataset5042_BRATS2016",
        "Dataset5043_BrainDevelopment",
        "Dataset5044_EPISURG",
        "Dataset5066_WMH",
        "Dataset5083_IXIT1",
        "Dataset5084_IXIT2",
        "Dataset5085_IXIPD",
        "Dataset6000_PPMI-T1-3T-PreProc",
        "Dataset6001_ADNI-group-T1-3T-PreProc",
        "Dataset6002_OASIS3",
    ]

    list_of_id_test_datasets = [
        # "Dataset4999_IXIAllModalities",
        "Dataset5083_IXIT1",
        "Dataset5084_IXIT2",
        "Dataset5085_IXIPD",
    ]

    list_of_ood_test_datasets = [
        "Dataset6003_AIBL",
    ]

    list_of_test_datasets = list_of_id_test_datasets + list_of_ood_test_datasets

    gigamed = GigaMedDataset()
    assert (
        gigamed.get_dataset_names_with_longitudinal(id=True)
        == datasets_with_longitudinal
    )
    assert (
        gigamed.get_dataset_names_with_multiple_modalities(id=True)
        == datasets_with_multiple_modalities
    )
    assert (
        gigamed.get_dataset_names_with_one_modality(id=True)
        == datasets_with_one_modality
    )


# if __name__ == "__main__":
#     train_dataset_names = [
#         "Dataset4999_IXIAllModalities",
#         "Dataset1000_PPMI",
#         "Dataset1001_PACS2019",
#         "Dataset1002_AIBL",
#         "Dataset1004_OASIS2",
#         "Dataset1005_OASIS1",
#         "Dataset1006_OASIS3",
#         "Dataset1007_ADNI",
#     ]

#     list_of_id_test_datasets = [
#         # "Dataset4999_IXIAllModalities",
#         "Dataset5083_IXIT1",
#         "Dataset5084_IXIT2",
#         "Dataset5085_IXIPD",
#     ]

#     list_of_ood_test_datasets = [
#         "Dataset6003_AIBL",
#     ]

#     list_of_test_datasets = list_of_id_test_datasets + list_of_ood_test_datasets

#     gigamed = GigaMedDataset()
#     # print(gigamed.get_dataset_names_with_longitudinal(id=True))
#     # print(gigamed.get_dataset_names_with_multiple_modalities(id=True))
#     # print(gigamed.get_dataset_names_with_one_modality(id=True))
#     # assert (
#     #     gigamed.get_dataset_names_with_longitudinal(id=True)
#     #     == datasets_with_longitudinal
#     # )
#     # assert (
#     #     gigamed.get_dataset_names_with_multiple_modalities(id=True)
#     #     == datasets_with_multiple_modalities
#     # )
#     # assert (
#     #     gigamed.get_dataset_names_with_one_modality(id=True)
#     #     == datasets_with_one_modality
#     # )
