import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def str_or_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def align_img(grid, x, mode="bilinear"):
    return F.grid_sample(
        x,
        grid=grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )


def displacement2pytorchflow(displacement_field, input_space="voxel"):
    """Converts displacement field in index coordinates into a flow-field usable by F.grid_sample.
    Assumes original space is in index (voxel) units, 256x256x256.
    Output will be in the [-1, 1] space.

    :param: displacement_field: (N, D, H, W, 3).
    """
    assert displacement_field.shape[-1] == 3, "Displacement field must have 3 channels"
    W, H, D = displacement_field.shape[1:-1]

    # Step 1: Create the original grid for 3D
    coords_x, coords_y, coords_z = torch.meshgrid(
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, D),
        indexing="ij",
    )
    coords = torch.stack([coords_z, coords_y, coords_x], dim=-1)  # Shape: (D, H, W, 3)
    coords = coords.unsqueeze(0).to(displacement_field)  # Shape: (N, 3, D, H, W), N=1

    # Step 2: Normalize the displacement field
    # Convert physical displacement values to the [-1, 1] range
    # Assuming the displacement field is given in voxel units (physical coordinates)
    if input_space == "voxel":
        for i, dim_size in enumerate(
            [W, H, D]
        ):  # Note the order matches x, y, z as per the displacement_field
            # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
            displacement_field[..., i] = 2 * displacement_field[..., i] / (dim_size - 1)

    # Step 3: Add the displacement field to the original grid to get the transformed coordinates
    return coords + displacement_field


def pytorchflow2displacement(flow):
    """Converts pytorch flow-field in [-1, 1] to a displacement field in index (voxel) units

    :param: flow: (N, D, H, W, 3).
    """
    flow = flow.permute(0, 4, 1, 2, 3)  # Bring channels to second dimension
    shape = flow.shape[2:]

    # Scale normalized flow to pixel indices
    for i in range(3):
        flow[:, i, ...] = (flow[:, i, ...] + 1) / 2 * (shape[i] - 1)

    # Create an image grid for the target size
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=0).unsqueeze(0).to(flow.device, dtype=torch.float32)

    # Calculate displacements from the image grid
    disp = flow - grid
    return disp


def rescale_intensity(array, out_range=(0, 1), percentiles=(0, 100)):
    if isinstance(array, torch.Tensor):
        array = array.float()

    if percentiles != (0, 100):
        cutoff = np.percentile(array, percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]
    in_min = array.min()
    in_range = array.max() - in_min
    out_min = out_range[0]
    out_range = out_range[1] - out_range[0]

    array -= in_min
    array /= in_range
    array *= out_range
    array += out_min
    return array


def sample_valid_coordinates(x, num_points, dim, point_space="norm", indexing="xy"):
    """
    x: input img, (1,1,dim1,dim2) or (1,1,dim1,dim2,dim3)
    num_points: number of points
    dim: Dimension, either 2 or 3

    Returns:
      points: Normalized coordinates in [0, 1], (1, num_points, dim)
    """
    if dim == 2:
        coords = sample_valid_coordinates_2d(x, num_points, point_space=point_space)
    elif dim == 3:
        coords = sample_valid_coordinates_3d(x, num_points, point_space=point_space)
    else:
        raise NotImplementedError
    if indexing == "ij":
        coords = coords.flip(-1)
    return coords


def sample_valid_coordinates_2d(x, num_points, point_space="norm"):
    eps = 0
    mask = x > eps
    indices = []
    for i in range(num_points):
        print(f"{i+1}/{num_points}")
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            sample[:, :, dim1, dim2] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                if point_space == "norm":
                    indices.append([dim2 / x.size(3), dim1 / x.size(2)])
                else:
                    indices.append([dim2, dim1])

    return torch.tensor(indices).view(1, num_points, 2)


def sample_valid_coordinates_3d(x, num_points, point_space="norm"):
    eps = 1e-1
    mask = x > eps
    indices = []
    for i in range(num_points):
        print(f"{i+1}/{num_points}")
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            dim3 = np.random.randint(0, x.size(4))
            sample[:, :, dim1, dim2, dim3] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                if point_space == "norm":
                    indices.append(
                        [dim3 / x.size(4), dim2 / x.size(3), dim1 / x.size(2)]
                    )
                else:
                    indices.append([dim3, dim2, dim1])

    return torch.tensor(indices).view(1, num_points, 3)


def one_hot_eval_synthseg(asegs):
    subset_regs = [
        [0, 24],  # Background and CSF
        [13, 52],  # Pallidum
        [18, 54],  # Amygdala
        [11, 50],  # Caudate
        [3, 42],  # Cerebral Cortex
        [17, 53],  # Hippocampus
        [10, 49],  # Thalamus
        [12, 51],  # Putamen
        [2, 41],  # Cerebral WM
        [8, 47],  # Cerebellum Cortex
        [4, 43],  # Lateral Ventricle
        [7, 46],  # Cerebellum WM
        [16, 16],  # Brain-Stem
    ]

    N, _, dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(N, chs, dim1, dim2, dim3)

    for i, s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1])
        one_hot[:, i, :, :, :] = (combined_vol * 1).float()

    mask = one_hot.sum(1).squeeze()
    ones = torch.ones_like(mask)
    non_roi = ones - mask
    one_hot[:, -1, :, :, :] = non_roi

    assert (
        one_hot.sum(1).sum() == N * dim1 * dim2 * dim3
    ), "One-hot encoding does not add up to 1"
    return one_hot


def one_hot(seg, num_classes=None):
    """Converts a segmentation to one-hot encoding.

    seg: (N, 1, D, H, W) tensor of integer labels
    """
    # Get unique labels and sort them
    unique_labels = torch.unique(seg)

    # Create a mapping from original labels to consecutive labels
    remap_dict = {label.item(): i for i, label in enumerate(unique_labels)}

    # Create a tensor for the remapped labels
    remapped_labels = seg.clone()
    for original_label, new_label in remap_dict.items():
        remapped_labels[seg == original_label] = new_label

    return F.one_hot(remapped_labels, num_classes=num_classes)[:, 0].permute(
        0, 4, 1, 2, 3
    )


def one_hot_subsampled(seg, subsample_num=14):
    onehot = one_hot(seg)
    selected_vals = np.random.choice(onehot.shape[1], subsample_num, replace=False)
    return onehot[:, selected_vals]


def one_hot_subsampled_pair(seg1, seg2, subsample_num=14):
    # Determine the unique integers in both segmentations
    unique_vals1 = np.unique(seg1.cpu().detach().numpy())
    unique_vals2 = np.unique(seg2.cpu().detach().numpy())

    # Take intersection
    unique_vals = np.intersect1d(unique_vals1, unique_vals2, assume_unique=True)

    # Subsample (if more than subsample_num values)
    if len(unique_vals) > subsample_num:
        selected_vals = np.random.choice(unique_vals, subsample_num, replace=False)
    else:
        selected_vals = unique_vals
        subsample_num = len(unique_vals)

    # Step 3: Create a mapping for the selected integers
    mapping = {val: i for i, val in enumerate(selected_vals)}

    # Step 4: Apply one-hot encoding to both segmentations with the mapping
    def apply_one_hot(asegs, mapping, subsample_num):
        one_hot_maps = torch.zeros(
            (asegs.shape[0], subsample_num, *asegs.shape[2:]),
            dtype=torch.float32,
            device=asegs.device,
        )
        for val, new_idx in mapping.items():
            one_hot_maps[:, new_idx] = (asegs == val).float()
        return one_hot_maps

    one_hot_maps1 = apply_one_hot(seg1, mapping, subsample_num)
    one_hot_maps2 = apply_one_hot(seg2, mapping, subsample_num)

    return one_hot_maps1, one_hot_maps2


def convert_points_norm2voxel(points, grid_sizes):
    """
    Rescale points from [-1, 1] to a uniform voxel grid with different sizes along each dimension.

    Args:
        points (bs, num_points, dim): Array of points in the normalized space [-1, 1].
        grid_sizes (bs, dim): Array of grid sizes for each dimension.

    Returns:
        Array of points in voxel space.
    """
    grid_sizes = torch.tensor(grid_sizes).to(points.device)
    assert grid_sizes.shape[-1] == points.shape[-1], "Dimensions don't match"
    translated_points = points + 1
    scaled_points = (translated_points * grid_sizes) / 2
    rescaled_points = scaled_points - 0.5
    return rescaled_points


def convert_points_voxel2norm(points, grid_sizes):
    """
    Reverse rescale points from a uniform voxel grid to the normalized space [-1, 1].

    Args:
        points (bs, num_points, dim): Array of points in the voxel space.
        grid_sizes (bs, dim): Array of grid sizes for each dimension.

    Returns:
        Array of points in the normalized space [-1, 1].
    """
    grid_sizes = torch.tensor(grid_sizes).to(points.device)
    assert grid_sizes.shape[-1] == points.shape[-1], "Dimensions don't match"
    rescaled_points_shifted = points + 0.5
    normalized_points = (2 * rescaled_points_shifted / grid_sizes) - 1
    return normalized_points


def convert_points_voxel2real(points, affine):
    """
    Convert points from uniform voxel grid to real world coordinates.

    Args:
        points (bs, num_points, dim): points in the normalized space [-1, 1].
        affine (bs, dim+1, dim+1): Square affine matrix
    """
    batch_size, num_points, _ = points.shape
    # Convert to homogeneous coordinates
    ones = torch.ones(batch_size, num_points, 1).to(points.device)
    points = torch.cat([points, ones], dim=2)

    # Apply the affine matrix
    real_world_points = torch.bmm(affine, points.permute(0, 2, 1)).permute(0, 2, 1)

    # Remove the homogeneous coordinate
    return real_world_points[:, :, :-1]


def convert_points_real2voxel(points, affine):
    """
    Convert points from uniform voxel grid to real world coordinates.

    Args:
        points (bs, num_points, dim): points in the normalized space [-1, 1].
        affine (bs, dim+1, dim+1): Square affine matrix
    """

    batch_size, num_points, _ = points.shape

    # Step 1: Convert to homogeneous coordinates by adding a column of ones
    ones = torch.ones(batch_size, num_points, 1).to(points.device)
    points = torch.cat([points, ones], dim=2)

    # Step 2: Compute the inverse affine matrices
    inverse_affine = torch.inverse(affine)

    # Step 3: Apply the inverse affine matrix
    points = torch.bmm(inverse_affine, points.permute(0, 2, 1)).permute(0, 2, 1)

    # Remove the homogeneous coordinate
    return points[:, :, :-1]


def convert_points_norm2real(points, affine_matrices, voxel_sizes):
    """
    Converts points from voxel coordinates (in the range [-1, 1]) to real world coordinates using batch-specific affine matrices.

    Args:
        points (torch.Tensor): The tensor of points with shape (batch_size, num_points, dimension).
        affine_matrices (torch.Tensor): The batch of affine matrices with shape (batch_size, dimension+1, dimension+1).
        voxel_sizes (torch.Tensor): The batch of voxel sizes with shape (batch_size, dimension).

    Returns:
        torch.Tensor: The points in real world coordinates with shape (batch_size, num_points, dimension).
    """
    denormalized_points = convert_points_norm2voxel(points, voxel_sizes)
    return convert_points_voxel2real(denormalized_points, affine_matrices)


def convert_points_real2norm(real_world_points, affine_matrices, voxel_sizes):
    """
    Converts points from real world coordinates to voxel coordinates (in the range [-1, 1]) using batch-specific affine matrices.

    Args:
        real_world_points (torch.Tensor): The tensor of real world points with shape (batch_size, num_points, dimension).
        affine_matrices (torch.Tensor): The batch of affine matrices with shape (batch_size, dimension+1, dimension+1).
        voxel_sizes (torch.Tensor): The batch of voxel sizes with shape (batch_size, dimension).

    Returns:
        torch.Tensor: The points in voxel coordinates with shape (batch_size, num_points, dimension).
    """
    voxel_points = convert_points_real2voxel(real_world_points, affine_matrices)
    return convert_points_voxel2norm(voxel_points, voxel_sizes)


def convert_flow_voxel2norm(flow, dim_sizes):
    """
    Parameters:
    - flow (torch.Tensor): The flow field tensor of shape (N, D, H, W, 3) in voxel coordinates.
    """
    # Convert physical displacement values to the [-1, 1] range
    # Assuming the displacement field is given in voxel units (physical coordinates)
    for i, dim_size in enumerate(
        dim_sizes
    ):  # Note the order matches x, y, z as per the displacement_field
        # Normalize such that the displacement of 1 full dimension length corresponds to a move from -1 to 1
        flow[..., i] = 2 * (flow[..., i] + 0.5) / dim_size - 1

    return flow


def uniform_voxel_grid(grid_shape, dim=3):
    if dim == 2:
        x = torch.arange(grid_shape[2])
        y = torch.arange(grid_shape[3])
        grid = torch.meshgrid(x, y, indexing="ij")
    else:
        x = torch.arange(grid_shape[2])
        y = torch.arange(grid_shape[3])
        z = torch.arange(grid_shape[4])
        grid = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack(grid, dim=-1).float()
    return grid


def uniform_norm_grid(grid_shape, dim=3):
    if dim == 2:
        x = torch.linspace(-1, 1, grid_shape[2])
        y = torch.linspace(-1, 1, grid_shape[3])
        grid = torch.meshgrid(x, y, indexing="ij")
    else:
        x = torch.linspace(-1, 1, grid_shape[2])
        y = torch.linspace(-1, 1, grid_shape[3])
        z = torch.linspace(-1, 1, grid_shape[4])
        grid = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack(grid, dim=-1).float()
    return grid


def random_masking(input_image, mask_ratio, patch_size, mask_val=0):
    """
    Performs random masking on a 3D input image, similar to masked autoencoders (MAE), and fills the masked regions with mask_val.

    Args:
    - input_image (torch.Tensor): Input image of shape (batch_size, num_ch, l, w, h).
    - mask_ratio (float): The ratio of patches to mask (e.g., 0.75 for 75% masking).
    - patch_size (tuple): Patch size in the form (pl, pw, ph) representing length, width, and height.
    - mask_val (float): The value to assign to the masked regions (default is 0).

    Returns:
    - masked_image (torch.Tensor): Image with patches randomly masked with `mask_val`.
    - mask (torch.Tensor): The binary mask applied to the input.
    """
    batch_size, num_ch, l, w, h = input_image.shape
    pl, pw, ph = patch_size

    # Ensure the dimensions are divisible by the patch size
    assert (
        l % pl == 0 and w % pw == 0 and h % ph == 0
    ), "Input dimensions must be divisible by the patch size."

    # Calculate the number of patches along each dimension
    num_patches_l = l // pl
    num_patches_w = w // pw
    num_patches_h = h // ph
    total_patches = num_patches_l * num_patches_w * num_patches_h

    # Create a binary mask with the same number of patches
    num_masked_patches = int(mask_ratio * total_patches)
    mask = torch.ones(batch_size, total_patches, device=input_image.device)

    for i in range(batch_size):
        # Randomly select patches to mask
        perm = torch.randperm(total_patches)[:num_masked_patches]
        mask[i, perm] = 0  # Set selected patches to zero (masked)

    # Reshape the mask to match the patch layout
    mask = mask.view(batch_size, num_patches_l, num_patches_w, num_patches_h)
    mask = mask.unsqueeze(1)  # Add a channel dimension to broadcast with input_image

    # Repeat the mask over the patch dimensions
    mask = F.interpolate(mask, size=(l, w, h), mode="nearest")

    # Apply the mask: keep unmasked values, and set masked regions to mask_val
    masked_image = input_image * mask + (1 - mask) * mask_val

    return masked_image, mask
