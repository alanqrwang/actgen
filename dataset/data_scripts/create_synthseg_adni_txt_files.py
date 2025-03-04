import pandas as pd
import numpy as np
import os
from glob import glob


def get_input_output_paths_from_metadata(root_path, metadata_csv_path):
    # Load metadata from CSV
    metadata_df = pd.read_csv(metadata_csv_path)

    # Create a list to hold torchio.Subject objects
    all_input_paths = []

    # Iterate over the metadata DataFrame
    for i, row in metadata_df.iterrows():
        print(f"{i}/{len(metadata_df)}")
        image_data_id = row["Image Data ID"]
        subject_id = row["Subject"]

        # Construct the expected directory structure based on the given format
        subject_dir = os.path.join(
            root_path, subject_id, "*", "*", image_data_id, "*.npy"
        )
        input_paths = glob(subject_dir)
        all_input_paths += input_paths

    print(f"Total input paths: {len(all_input_paths)}")
    return all_input_paths


def convert_npy_to_npz_with_affine(paths, destination_directory):
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Define the affine matrix (identity matrix)
    affine_matrix = np.eye(4)  # 4x4 identity matrix

    # Convert each .npy file to a .npz file with an affine matrix and save in the destination directory
    for path in paths:
        # Ensure the .npy file exists
        if os.path.isfile(path):
            # Load the .npy file
            data = np.load(path)

            # Get the base file name (without .npy extension) and create the new .npz file name
            base_name = os.path.basename(path).replace(".npy", "")
            dest_path = os.path.join(destination_directory, f"{base_name}.npz")

            # Save the data and affine matrix as .npz
            np.savez(dest_path, vol_data=data, affine=affine_matrix)
            print(f"Converted and saved {path} as {dest_path} with an affine matrix")
        else:
            print(f"File not found: {path}")


def get_all_file_paths(directory):
    # List to store file paths
    file_paths = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.startswith("synthseg_"):
                # Create the full path for each file and add it to the list
                file_paths.append(os.path.join(root, file))

    return file_paths


# root_path = (
#     "/simurgh/u/alanqw/data/from_sherlock/scratch/groups/eadeli/data/stru/t1/adni"
# )
# metadata_csv_path = "/simurgh/u/alanqw/data/from_sherlock/scratch/groups/eadeli/data/stru/t1/metadata/adni/preprocessed_ADNI_mri_2_14_2024.csv"
# subject_paths = get_input_output_paths_from_metadata(root_path, metadata_csv_path)


destination_directory = "/simurgh/u/alanqw/data/adni_npy_to_npz/"

# convert_npy_to_npz_with_affine(subject_paths, destination_directory)


# Create a new list of output paths where the file name is prepended with 'synthseg_'
input_paths = get_all_file_paths(destination_directory)
output_paths = []
for path in input_paths:
    # Get the directory and file name
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)

    # Prepend 'synthseg_' to the file name
    new_file_name = f"synthseg_{file_name}"

    # Join the directory name and the new file name to get the new path
    new_path = os.path.join(dir_name, new_file_name)

    # Append the new path to the list
    output_paths.append(new_path)

# **Added Section: Filter out paths where the output file already exists**
filtered_input_paths = []
filtered_output_paths = []

print(len(filtered_input_paths))
print(len(filtered_output_paths))
for i, (in_path, out_path) in enumerate(zip(input_paths, output_paths)):
    if not os.path.exists(out_path):
        filtered_input_paths.append(in_path)
        filtered_output_paths.append(out_path)
        print("appending", i)
    # else:
    #     print(f"Output file already exists and will be skipped: {out_path}")
print(len(filtered_input_paths))
print(len(filtered_output_paths))

# Specify the filename for the txt file
input_filename = "synthseg_adni_filtered_input_paths.txt"
output_filename = "synthseg_adni_filtered_output_paths.txt"

with open(input_filename, "w") as f:
    for path in filtered_input_paths:
        f.write(f"{path}\n")

print(f"Input paths saved to {input_filename}")

with open(output_filename, "w") as f:
    for path in filtered_output_paths:
        f.write(f"{path}\n")

print(f"Output paths saved to {output_filename}")
