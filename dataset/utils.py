import random
import itertools
import torchio as tio
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from itertools import combinations


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
    Given pair of subject lists, samples pairs of subjects without restriction."""

    def __init__(
        self,
        subject_pairs_list,
        transform=None,
    ):
        super().__init__()
        self.subject_list = subject_pairs_list
        self.transform = transform

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, i):
        sub1, sub2 = self.subject_list[i]
        sub1.load()
        sub2.load()
        if self.transform:
            sub1 = self.transform(sub1)
            sub2 = self.transform(sub2)
        return sub1, sub2


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


class RandomAggregatedDataset(Dataset):
    """Aggregates multiple datasets and returns random samples from them."""

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        chosen_dataset = random.choice(self.datasets)
        return chosen_dataset[i]


class KeyMorphDataset:
    def _parse_test_mod(self, mod):
        if isinstance(mod, str):
            mod1, mod2 = mod.split("_")
        else:
            mod1, mod2 = mod
        return mod1, mod2

    def get_subjects(self, train):
        raise NotImplementedError

    def get_reference_subject(self, train=True):
        """Get a reference subject with all modalities"""
        subjects = self.get_subjects(train)

    def get_pretrain_loader(self, batch_size, num_workers, transform):
        subjects = self.get_subjects(
            train=True,
        )
        if isinstance(subjects, dict):
            pretrain_datasets = [
                tio.data.SubjectsDataset(
                    subjects_list,
                    transform=transform,
                )
                for subjects_list in subjects.values()
            ]
            pretrain_dataset = ConcatDataset(pretrain_datasets)
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

    def get_train_loader(self, batch_size, num_workers, mix_modalities, transform):
        subjects = self.get_subjects(
            train=True,
        )
        if isinstance(subjects, dict):
            train_mods = list(subjects.keys())
            if mix_modalities:
                mod_pairs = list(combinations(train_mods, 2))
            else:
                mod_pairs = [(m, m) for m in train_mods]

            paired_datasets = []
            for mod1, mod2 in mod_pairs:
                paired_datasets.append(
                    PairedDataset(
                        list(itertools.product(subjects[mod1], subjects[mod2])),
                        transform=transform,
                    )
                )
            train_dataset = ConcatDataset(paired_datasets)
        else:
            train_dataset = PairedDataset(
                list(zip(subjects[0], subjects[1])),
                transform=transform,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        return train_loader

    def get_test_loaders(self, batch_size, num_workers, transform, list_of_mods):
        subjects = self.get_subjects(
            train=False,
        )
        if isinstance(subjects, dict):
            test_datasets = []
            for dataset_name in list_of_mods:
                mod1, mod2 = self._parse_test_mod(dataset_name)
                subjects1 = subjects[mod1]
                subjects2 = subjects[mod2]
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
            self.get_train_loader(batch_size, num_workers, mix_modalities, transform),
            self.get_test_loaders(
                batch_size, num_workers, transform, list_of_test_mods
            ),
        )


class RSSLDataset:
    def _parse_test_mod(self, mod):
        if isinstance(mod, str):
            mod1, mod2 = mod.split("_")
        else:
            mod1, mod2 = mod
        return mod1, mod2

    def get_subjects(self, train):
        raise NotImplementedError

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

    def get_pretrain_loader(self, batch_size, num_workers, transform):
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

    def get_train_loader(self, batch_size, num_workers, transform):
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

    def get_test_loaders(self, batch_size, num_workers, transform, list_of_mods):
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
        return dataset[i], family_name

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
