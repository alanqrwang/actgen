import nibabel as nib
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.ndimage import morphology
from sklearn.metrics import accuracy_score, roc_auc_score


class MSELoss(torch.nn.Module):
    """MSE loss."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


def masked_mse_loss(image1, image2, mask):
    """
    Compute the mean-squared-error (MSE) between two images only at the unmasked regions.

    Args:
    - image1 (torch.Tensor): The first input image of shape (batch_size, num_ch, l, w, h).
    - image2 (torch.Tensor): The second input image of the same shape as image1.
    - mask (torch.Tensor): The binary mask with the same spatial dimensions as image1, where 1 = unmasked, 0 = masked.

    Returns:
    - mse_loss (torch.Tensor): The mean-squared error computed over unmasked regions.
    """
    # Ensure the two images have the same shape
    assert image1.shape == image2.shape, "Input images must have the same shape"

    # Compute squared differences between the two images
    squared_diff = (image1 - image2) ** 2

    # Apply the mask to keep only the unmasked regions (where mask == 1)
    unmasked_squared_diff = squared_diff * mask

    # Compute the sum of squared differences over the unmasked regions
    sum_squared_diff = unmasked_squared_diff.sum()

    # Count the number of unmasked elements
    num_unmasked = mask.sum()

    # Avoid division by zero in case there are no unmasked regions
    if num_unmasked == 0:
        return torch.tensor(0.0, device=image1.device)

    # Compute the mean squared error over the unmasked regions
    mse_loss = sum_squared_diff / num_unmasked
    return mse_loss


class SpatialCovarianceLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        """
        Initializes the SpatialCovarianceLoss module.

        Args:
            lambda_cov (float): Scaling factor for the covariance loss.
            reduction (str): Specifies the reduction to apply to the output:
                             'mean' | 'sum'. 'mean': the output will be averaged over the batch.
                             'sum': the outputs will be summed over the batch.
        """
        super(SpatialCovarianceLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("Reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, x):
        """
        Computes the covariance loss for a batch of spatial representations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_ch, l, w, h).

        Returns:
            torch.Tensor: Scalar tensor representing the covariance loss.
        """
        batch_size, n_ch, l, w, h = x.shape
        device = x.device

        # Reshape x to (batch_size, n_ch, total_positions)
        x = x.view(
            batch_size, n_ch, -1
        )  # Now x has shape (batch_size, n_ch, total_positions)

        # Zero-center the data for each sample
        x = x - x.mean(dim=2, keepdim=True)  # Subtract mean over spatial positions

        # Compute covariance matrices for each sample
        # x is (batch_size, n_ch, total_positions)
        # We need to compute (n_ch, n_ch) covariance matrix for each sample
        num_positions = x.shape[2]

        # Compute covariance matrices in a batched manner
        cov = torch.matmul(x, x.transpose(1, 2)) / (
            num_positions - 1
        )  # Shape: (batch_size, n_ch, n_ch)

        # Create a mask to select off-diagonal elements
        eye = torch.eye(n_ch, device=device).bool()  # Shape: (n_ch, n_ch)
        off_diag_mask = ~eye  # Invert the mask

        # Extract off-diagonal elements and compute loss per sample
        off_diag_elements = cov[
            :, off_diag_mask
        ]  # Shape: (batch_size, n_ch * (n_ch - 1))
        cov_loss_per_sample = (off_diag_elements**2).sum(
            dim=1
        ) / n_ch  # Shape: (batch_size,)

        # Aggregate the losses over the batch
        if self.reduction == "mean":
            cov_loss = cov_loss_per_sample.mean()
        else:  # 'sum'
            cov_loss = cov_loss_per_sample.sum()

        return cov_loss


class DiceLoss(torch.nn.Module):
    """Dice loss (lower is better).

    Supports 2d or 3d inputs.
    Supports hard dice or soft dice.
    If soft dice, returns scalar dice loss for entire slice/volume.
    If hard dice, returns:
        total_avg: scalar dice loss for entire slice/volume ()
        regions_avg: dice loss per region (n_ch)
        ind_avg: dice loss per region per pixel (bs, n_ch)

    """

    def __init__(self, hard=False, return_regions=False):
        super(DiceLoss, self).__init__()
        self.hard = hard
        self.return_regions = return_regions

    def forward(self, pred, target, ign_first_ch=False):
        eps = 1
        assert pred.size() == target.size(), "Input and target are different dim"

        if len(target.size()) == 4:
            n, c, _, _ = target.size()
        if len(target.size()) == 5:
            n, c, _, _, _ = target.size()

        target = target.contiguous().view(n, c, -1)
        pred = pred.contiguous().view(n, c, -1)

        if self.hard:  # hard Dice
            pred_onehot = torch.zeros_like(pred)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            pred = torch.scatter(pred_onehot, 1, pred, 1.0)
        if ign_first_ch:
            target = target[:, 1:, :]
            pred = pred[:, 1:, :]

        num = torch.sum(2 * (target * pred), 2) + eps
        den = (pred * pred).sum(2) + (target * target).sum(2) + eps
        dice_loss = 1 - num / den
        total_avg = torch.mean(dice_loss)
        regions_avg = torch.mean(dice_loss, 0)

        if self.return_regions:
            return regions_avg
        else:
            return total_avg


def fast_dice(x, y):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on
    :return: numpy array with Dice scores in the same order as labels.
    """

    x = x.argmax(1)
    y = y.argmax(1)
    labels = np.unique(np.concatenate([np.unique(x), np.unique(y)]))
    assert (
        x.shape == y.shape
    ), "both inputs should have same size, had {} and {}".format(x.shape, y.shape)

    if len(labels) > 1:
        # sort labels
        labels_sorted = np.sort(labels)

        # build bins for histograms
        label_edges = np.sort(
            np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1])
        )
        label_edges = np.insert(
            label_edges,
            [0, len(label_edges)],
            [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1],
        )

        # compute Dice and re-arrange scores in initial order
        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = (
            2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        )
        dice_score = dice_score[np.searchsorted(labels_sorted, labels)]

    else:
        dice_score = dice(x == labels[0], y == labels[0])

    return np.mean(dice_score)  # Remove mean to get region-level scores


def dice(x, y):
    """Implementation of dice scores for 0/1 numpy array"""
    return 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))


def _check_type(t):
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    return t


# Hausdorff Distance
def _surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(bool))
    input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = (
        input_1.astype(int) - morphology.binary_erosion(input_1, conn).astype(int)
    ).astype(bool)
    Sprime = (
        input_2.astype(int) - morphology.binary_erosion(input_2, conn).astype(int)
    ).astype(bool)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds


def hausdorff_distance(test_seg, gt_seg):
    """Computes Hausdorff distance on brain surface.

    Assumes segmentation maps are one-hot and first channel is background.

    Args:
        test_seg: Test segmentation map (bs, n_ch, l, w, h)
        gt_seg: Ground truth segmentation map (bs, n_ch, l, w, h)
    """
    test_seg = _check_type(test_seg)
    gt_seg = _check_type(gt_seg)

    hd = 0
    for i in range(len(test_seg)):
        hd += _surfd(test_seg[i, 0], gt_seg[i, 0], [1.25, 1.25, 10], 1).max()
    return hd / len(test_seg)


# Jacobian determinant
def _jacobian_determinant(disp):
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradz_disp, grady_disp, gradx_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    return jacdet


def jdstd(disp):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    return jd.std()


def jdlessthan0(disp, as_percentage=False):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    if as_percentage:
        return np.count_nonzero(jd <= 0) / len(jd.flatten())
    return np.count_nonzero(jd <= 0)


class LC2(torch.nn.Module):
    def __init__(self, radiuses=(3, 5, 7)):
        super(LC2, self).__init__()
        self.radiuses = radiuses
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    def forward(self, us, mr):
        s = self.run(us, mr, self.radiuses[0])
        for r in self.radiuses[1:]:
            s += self.run(us, mr, r)
        return s / len(self.radiuses)

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)
        assert us.shape == mr.shape
        assert us.shape[1] == us.shape[2] == us.shape[3]
        assert us.shape[1] % 2 == 1, "Input must be odd size"

        bs = mr.size(0)
        pad = (mr.size(1) - (2 * radius + 1)) // 2
        count = (2 * radius + 1) ** 3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = (
            torch.einsum("bip,bjp->bij", A, A) / count
            + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        )
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1) ** 2
        dist = (
            torch.mean(b**2, dim=1)
            + torch.einsum("bi,bj,bij->b", coeff, coeff, C)
            - 2 * torch.einsum("bi,bi->b", coeff, Atb)
        )
        sym = (var - dist) / var.clamp_min(beta)

        return sym.clamp(0, 1)


class ImageLC2(torch.nn.Module):
    def __init__(self, patch_size=51, radiuses=(5,), reduction="mean"):
        super(ImageLC2, self).__init__()
        self.patch_size = patch_size
        self.radii = radiuses
        assert reduction in ["mean", None]
        self.reduction = reduction
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    @staticmethod
    def patch2batch(x, size, stride):
        """Converts image x into patches, then reshapes into batch of patches"""
        nch = x.shape[1]
        if len(x.shape) == 4:
            patches = x.unfold(2, size, stride).unfold(3, size, stride)
            return patches.reshape(-1, nch, size, size)
        else:
            patches = (
                x.unfold(2, size, stride)
                .unfold(3, size, stride)
                .unfold(4, size, stride)
            )
            return patches.reshape(-1, nch, size, size, size)

    def forward(self, us, mr):
        assert (
            us.shape == mr.shape
        ), f"Input and target have different shapes, {us.shape} vs {mr.shape}"
        assert (
            us.shape[-1] == us.shape[-2] == us.shape[-3]
        ), f"Dimensions must be equal, currently {us.shape}"
        assert us.shape[1] % 2 == 1, f"Input must be odd size, currently {us.shape}"

        # Convert to batch of patches
        r = self.radii[0]
        us_patch = self.patch2batch(us, self.patch_size, self.patch_size)
        mr_patch = self.patch2batch(mr, self.patch_size, self.patch_size)
        s = self.run(us_patch, mr_patch, r)
        for r in self.radii[1:]:
            us_patch = self.patch2batch(us, self.patch_size, self.patch_size)
            mr_patch = self.patch2batch(mr, self.patch_size, self.patch_size)
            s += self.run(us_patch, mr_patch, r)

        s = s / len(self.radii)
        if self.reduction == "mean":
            return s.mean()
        else:
            return s

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)

        bs = mr.size(0)
        pad = (mr.size(1) - (2 * radius + 1)) // 2
        count = (2 * radius + 1) ** 3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = (
            torch.einsum("bip,bjp->bij", A, A) / count
            + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        )
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1) ** 2
        dist = (
            torch.mean(b**2, dim=1)
            + torch.einsum("bi,bj,bij->b", coeff, coeff, C)
            - 2 * torch.einsum("bi,bi->b", coeff, Atb)
        )
        sym = (var - dist) / var.clamp_min(beta)

        return sym.clamp(0, 1)


# class LesionPenalty(torch.nn.Module):
#     def forward(self, weights, points_f, points_m, lesion_mask_f, lesion_mask_m):
#         ind_in_mask_f = ind_in_lesion(points_f, lesion_mask_f)
#         ind_in_mask_m = ind_in_lesion(points_m, lesion_mask_m)

#         gt = torch.ones_like(weights)
#         gt[ind_in_mask_f] = 0
#         gt[ind_in_mask_m] = 0

#         return F.mse_loss(gt, weights)


def _load_file(path):
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        return torch.tensor(nib.load(path).get_fdata())
    elif path.endswith(".npy"):
        return torch.tensor(np.load(path))
    else:
        raise ValueError("File format not supported")


class _AvgPairwiseLoss(torch.nn.Module):
    """Pairwise loss."""

    def __init__(self, metric_fn):
        super().__init__()
        self.metric_fn = metric_fn

    def forward(self, batch_of_imgs):
        loss = 0
        num = 0
        for i in range(len(batch_of_imgs)):
            for j in range(i + 1, len(batch_of_imgs)):
                if isinstance(batch_of_imgs[0], str):
                    img1 = _load_file(batch_of_imgs[i])
                    img2 = _load_file(batch_of_imgs[j])
                else:
                    img1 = batch_of_imgs[i : i + 1]
                    img2 = batch_of_imgs[j : j + 1]
                loss += self.metric_fn(img1, img2)
                num += 1
        return loss / num


class MSEPairwiseLoss(_AvgPairwiseLoss):
    """MSE pairwise loss."""

    def __init__(self):
        super().__init__(MSELoss().forward)


class SoftDicePairwiseLoss(_AvgPairwiseLoss):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(DiceLoss().forward)


class HardDicePairwiseLoss(_AvgPairwiseLoss):
    """Hard Dice pairwise loss."""

    def __init__(self):
        super().__init__(DiceLoss(hard=True).forward)


class HausdorffPairwiseLoss(_AvgPairwiseLoss):
    """Hausdorff pairwise loss."""

    def __init__(self):
        super().__init__(hausdorff_distance)


class _AvgGridMetric(torch.nn.Module):
    """Aggregated average metric for grids."""

    def __init__(self, metric_fn):
        super().__init__()
        self.metric_fn = metric_fn

    def forward(self, batch_of_grids):
        tot_jdstd = 0
        for i in range(len(batch_of_grids)):
            if isinstance(batch_of_grids[i], str):
                grid = _load_file(batch_of_grids[i])
            else:
                grid = batch_of_grids[i : i + 1]
            grid_permute = grid.permute(0, 4, 1, 2, 3)
            tot_jdstd += self.metric_fn(grid_permute)
        return tot_jdstd / len(batch_of_grids)


class AvgJDStd(_AvgGridMetric):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(jdstd)


class AvgJDLessThan0(_AvgGridMetric):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(jdlessthan0)


class MultipleAvgSegPairwiseMetric(torch.nn.Module):
    """Evaluate multiple pairwise losses on a batch of images,
    so that we don't need to load the images into memory multiple times."""

    def __init__(self):
        super().__init__()
        self.name2fn = {
            "dice": fast_dice,
            "harddice": DiceLoss(hard=True).forward,
            "harddiceroi": DiceLoss(hard=True, return_regions=True).forward,
            "softdice": DiceLoss().forward,
            "hausd": hausdorff_distance,
        }

    def forward(self, batch_of_imgs, fn_names):
        res = {name: 0 for name in fn_names}
        num = 0
        for i in range(len(batch_of_imgs)):
            for j in range(i + 1, len(batch_of_imgs)):
                if isinstance(batch_of_imgs[0], str):
                    img1 = _load_file(batch_of_imgs[i])
                    img2 = _load_file(batch_of_imgs[j])
                else:
                    img1 = batch_of_imgs[i : i + 1]
                    img2 = batch_of_imgs[j : j + 1]
                for name in fn_names:
                    res[name] += self.name2fn[name](img1, img2)
                num += 1
        return {name: res[name] / num for name in fn_names}


class MultipleAvgGridMetric(torch.nn.Module):
    """Evaluate multiple grid metrics on a batch of grids, mostly
    so that we don't need to load them into memory multiple times."""

    def __init__(self):
        super().__init__()
        self.name2fn = {
            "jdstd": jdstd,
            "jdlessthan0": jdlessthan0,
        }

    def forward(self, batch_of_grids, fn_names):
        res = {name: 0 for name in fn_names}
        for i in range(len(batch_of_grids)):
            if isinstance(batch_of_grids[i], str):
                grid = _load_file(batch_of_grids[i])
            else:
                grid = batch_of_grids[i : i + 1]
            grid_permute = grid.permute(0, 4, 1, 2, 3)
            for name in fn_names:
                res[name] += self.name2fn[name](grid_permute)
        return {name: res[name] / len(batch_of_grids) for name in fn_names}


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [
                *range(d - 1, d + 1),
                *reversed(range(1, d - 1)),
                0,
                *range(d + 1, ndims + 2),
            ]
            df[i] = dfi.permute(r)

        return df

    def forward(self, _, y_pred):
        if self.penalty == "l1":
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == "l2", (
                "penalty can only be l1 or l2. Got: %s" % self.penalty
            )
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


def check_type(x):
    return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x


def acc(pred, targets):
    """Returns accuracy given batch of categorical predictions and targets."""
    pred = check_type(pred)
    targets = check_type(targets)
    return accuracy_score(targets, pred)


class DINOLoss(torch.nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
        device="cuda",
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        # Initialize a center for teacher outputs to avoid collapse
        self.register_buffer("center", torch.zeros(1, out_dim).to(device))

    def forward(self, student_output, teacher_output):
        """
        student_output: NxC student logits
        teacher_output: NxC teacher logits (moving average of the student)
        """

        # Normalize the teacher's output by subtracting the center
        teacher_output = (teacher_output - self.center) / self.teacher_temp
        teacher_output = F.softmax(teacher_output, dim=-1)

        # Normalize the student's output
        student_output = student_output / self.student_temp
        student_output = F.log_softmax(student_output, dim=-1)

        # DINO loss (cross-entropy between student and teacher)
        loss = -torch.sum(teacher_output * student_output, dim=-1).mean()

        # Update the center for the teacher (moving average of the teacher's outputs)
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        # Update the center to avoid teacher collapse
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
