import numpy as np


def overlap_metrics(prediction, reference, voxel_ml=0.001):
    prediction, reference = prediction.astype(bool), reference.astype(bool)
    tp = np.count_nonzero(prediction & reference)
    fp = np.count_nonzero(prediction & ~reference)
    fn = np.count_nonzero(~prediction & reference)
    denominator = 2 * tp + fp + fn
    return {
        "dice": 2 * tp / denominator if denominator > 0 else 1.0,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 1.0,
        "fp_volume_ml": fp * voxel_ml,
        "fn_volume_ml": fn * voxel_ml,
    }


def hd95(prediction, reference, spacing=(1.0, 1.0, 1.0)):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    prediction, reference = prediction.astype(bool), reference.astype(bool)
    if not prediction.any() or not reference.any():
        return float("nan")
    prediction_surface = prediction & ~binary_erosion(prediction)
    reference_surface = reference & ~binary_erosion(reference)
    to_reference = distance_transform_edt(~reference_surface, sampling=spacing)[prediction_surface]
    to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)[reference_surface]
    return float(max(np.percentile(to_reference, 95), np.percentile(to_prediction, 95)))


def case_metrics(prediction, reference, spacing=(1.0, 1.0, 1.0)):
    voxel_ml = float(np.prod(spacing)) / 1000
    return {
        **overlap_metrics(prediction, reference, voxel_ml),
        "hd95_mm": hd95(prediction, reference, spacing),
        "pred_voxels": int(np.count_nonzero(prediction)),
        "ref_voxels": int(np.count_nonzero(reference)),
    }
