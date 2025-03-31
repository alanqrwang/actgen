import os
from glob import glob
import subprocess
import ants
import nibabel as nib
from bids import BIDSLayout

# === CONFIGURATION ===
BIDS_DIR = "/simurgh/group/ACT/data/bids/"  # Update this to your dataset path
MNI_TEMPLATE = "/simurgh/u/alanqw/mni_icbm152_nlin_asym_09a_nifti/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a_masked.nii"  # Ensure this file is downloaded

# Load BIDS dataset
layout = BIDSLayout(BIDS_DIR)
print(layout)

# Find all T1-weighted images
# t1w_files = layout.get(suffix='T1w', extension=['nii', 'nii.gz'], return_type='file')
# t1w_files = glob(BIDS_DIR + "*/*/sub-*/ses-*/anat/*_T1w.nii.gz")
# print(f"Found {len(t1w_files)} T1w images.")
flair_files = glob(BIDS_DIR + "*/*/sub-*/ses-*/anat/*_FLAIR.nii.gz")
print(f"Found {len(flair_files)} FLAIR images.")

files = flair_files

# === PROCESSING EACH SUBJECT ===
for i, t1w_file in enumerate(files):
    print(f"\n\n\nPROCESSING {i+1}/{len(files)}")
    # Extract subject ID and directory
    subject = layout.parse_file_entities(t1w_file)["subject"]
    subject_dir = os.path.dirname(t1w_file)  # Directory where original file is stored
    base_name = os.path.basename(t1w_file).replace(
        ".nii.gz", ""
    )  # Get filename without extension

    print(f"Processing subject: {subject}, saving outputs in: {subject_dir}")

    # Define output filenames with the same base name
    bet_output = os.path.join(subject_dir, f"{base_name}_brain.nii.gz")
    n4_output = os.path.join(subject_dir, f"{base_name}_n4.nii.gz")
    mni_output = os.path.join(subject_dir, f"{base_name}_mni_ants_affine.nii.gz")

    # Step 1: Skull Stripping with HD-BET
    if not os.path.exists(bet_output):
        print(f"Running HD-BET for subject {subject}...")
        subprocess.run(["hd-bet", "-i", t1w_file, "-o", bet_output])
    else:
        print(f"Skipping HD-BET, output already exists: {bet_output}")

    # Step 2: Intensity Normalization (Bias Field Correction) with ANTs
    if not os.path.exists(n4_output):
        print(f"Running N4 bias correction for subject {subject}...")
        brain = ants.image_read(bet_output)
        brain_n4 = ants.n4_bias_field_correction(brain)
        ants.image_write(brain_n4, n4_output)
    else:
        print(f"Skipping N4, output already exists: {n4_output}")

    # Step 3: Register to MNI Space using ANTs
    if not os.path.exists(mni_output):
        print(f"Affine registering subject {subject} to MNI space...")
        brain_n4 = ants.image_read(n4_output)
        template = ants.image_read(MNI_TEMPLATE)
        reg = ants.registration(
            fixed=template, moving=brain_n4, type_of_transform="Affine"
        )
        ants.image_write(reg["warpedmovout"], mni_output)
    else:
        print(f"Skipping MNI, output already exists: {mni_output}")

    print(f"Subject {subject} processing complete. Outputs saved in: {subject_dir}")
    print(f"  - Skull-stripped: {bet_output}")
    print(f"  - Bias-corrected: {n4_output}")
    print(f"  - MNI-registered: {mni_output}")

print("Processing complete for all subjects!")
