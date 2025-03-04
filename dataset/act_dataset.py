import os
import glob
import torchio as tio
from collections import defaultdict
from pprint import pprint
import pandas as pd

GROUP_MAP = {
    "Stretching and MLA": 0,
    "Cycling with SOP": 1,
    "Cycling ONLY": 2,
    "SOP ONLY": 3,
}
GROUP_LIST = [
    "Stretching and MLA",
    "Cycling with SOP",
    "Cycling ONLY",
    "SOP ONLY",
]


def get_torchio_subjects(bids_root, demographics_csv):
    """
    Parses the BIDS directory and returns a list of torchio.Subject objects,
    where each subject contains all their visits.
    """
    df = pd.read_csv(demographics_csv)
    t1_images = glob.glob(
        os.path.join(bids_root, "*", "*", "sub-*", "ses-*", "anat", "*_T1w.nii.gz")
    )

    subject_dict = defaultdict(dict)
    for img_path in t1_images:
        parts = img_path.split(os.sep)
        visit_name = parts[-6]  # Extract visit name (e.g., T1_baseline)
        subject_id = parts[-4]  # Extract sub-XXXX
        session_id = parts[-3]  # Extract ses-XX

        if subject_id not in subject_dict:
            subject_dict[subject_id] = {}
        subject_dict[subject_id][f"{visit_name}_{session_id}"] = tio.ScalarImage(
            img_path
        )

    subjects = []
    for id, visits in subject_dict.items():
        if "T1_baseline_ses-01" not in visits:
            continue
        visits["subject_id"] = id
        try:
            # Remove the 'sub-' prefix and convert to an integer to remove leading zeros
            numeric_id = int(id.replace("sub-", ""))
            # Build the csv id
            csv_id = f"s{numeric_id}"
            group_desc = df.loc[df["screening_id"] == csv_id, "group_desc"].values[0]
        except Exception as e:
            print(f"Error processing subject {id}: {e}")

        visits["group"] = GROUP_MAP[group_desc]
        pprint(visits.keys())
        subjects.append(tio.Subject(**visits))

        # --- Statistics Printing ---
    # Initialize counters for groups and visits
    group_counts = defaultdict(int)
    visit_counts = defaultdict(int)
    # Nested dictionary for group-specific visit counts: group -> visit count -> number of subjects
    group_visit_counts = defaultdict(lambda: defaultdict(int))

    for subj in subjects:
        group = GROUP_LIST[subj["group"]]
        group_counts[group] += 1

        # Count the number of visits (exclude 'subject_id' and 'group' keys)
        n_visits = len([k for k in subj.keys() if k not in ("subject_id", "group")])
        visit_counts[n_visits] += 1
        group_visit_counts[group][n_visits] += 1

    # Print the number of subjects per group
    print("Number of subjects per group:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}")

    # Print the overall number of subjects with at least 1, 2, 3, and 4 visits
    print("Number of subjects by visit count (overall) [at least]:")
    for threshold in range(1, 5):
        at_least = sum(count for n, count in visit_counts.items() if n >= threshold)
        print(f"  Subjects with at least {threshold} visit(s): {at_least}")

    # Print the number of subjects within each group by visit count (at least)
    print("Subjects by visit count within each group [at least]:")
    for group, visits_dict in group_visit_counts.items():
        print(f"Group: {group}")
        for threshold in range(1, 5):
            at_least = sum(count for n, count in visits_dict.items() if n >= threshold)
            print(f"  Subjects with at least {threshold} visit(s): {at_least}")
    # --- End of Statistics Printing ---
    return subjects


class ACTDataset:
    def __init__(
        self, root_bids_path, demo_csv, batch_size=8, num_workers=4, shuffle=True
    ):
        """
        Initialize the ADNI dataset class.

        Parameters:
        - subjects: list of torchio.Subject objects
        - batch_size: number of samples per batch
        - num_workers: number of subprocesses to use for data loading
        - shuffle: whether to shuffle the dataset before splitting
        """
        self.subjects = get_torchio_subjects(root_bids_path, demo_csv)

        split_idx_1 = int(0.8 * len(self.subjects))
        split_idx_2 = int(0.9 * len(self.subjects))
        self.train_subjects = self.subjects[:split_idx_1]
        self.val_subjects = self.subjects[split_idx_1:split_idx_2]
        self.test_subjects = self.subjects[split_idx_2:]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seg_available = False

    def get_train_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the training set.

        Returns:
        - train_loader: DataLoader for the training set
        """
        train_dataset = tio.SubjectsDataset(self.train_subjects, transform=transform)
        train_loader = tio.SubjectsLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.shuffle,
        )
        return train_loader

    def get_val_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the validation set.

        Returns:
        - val_loader: DataLoader for the validation set
        """
        val_dataset = tio.SubjectsDataset(self.val_subjects, transform=transform)
        val_loader = tio.SubjectsLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return val_loader

    def get_test_loader(self, batch_size, num_workers, transform=None):
        """
        Get the DataLoader for the test set.

        Returns:
        - test_loader: DataLoader for the test set
        """
        test_dataset = tio.SubjectsDataset(self.test_subjects, transform=transform)
        test_loader = tio.SubjectsLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        return test_loader


if __name__ == "__main__":
    dataset = ACTDataset(
        "/simurgh/group/ACT/data/bids", "/simurgh/group/ACT/demographics.csv"
    )
    train_loader = dataset.get_train_loader(8, 4)
    val_loader = dataset.get_val_loader(8, 4)
    test_loader = dataset.get_test_loader(8, 4)

    print(f"Train: {len(train_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    print(f"Test: {len(test_loader)} batches")

    for batch in train_loader:
        print(batch)
        break

    for batch in val_loader:
        print(batch)
        break

    for batch in test_loader:
        print(batch)
        break
