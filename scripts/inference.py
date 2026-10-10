import numpy as np
import torch
from scipy.ndimage import label


def window_starts(size, patch, stride):
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if starts[-1] + patch < size:
        starts.append(size - patch)
    return starts


def gaussian_weights(patch_size, sigma_scale=0.25):
    axes = [np.arange(p) - (p - 1) / 2 for p in patch_size]
    grid = np.meshgrid(*axes, indexing="ij")
    weights = np.exp(-sum(g ** 2 / (2 * (s * sigma_scale) ** 2) for g, s in zip(grid, patch_size)))
    return weights.astype(np.float32)


@torch.no_grad()
def predict_volume(model, image, device, patch_size=(64, 64, 64), stride=(32, 32, 32), batch_size=8):
    model.eval()
    image = torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(3, 0, 1, 2)))  # [C, X, Y, Z]
    shape = image.shape[1:]
    weights = torch.from_numpy(gaussian_weights(patch_size)).to(device)
    prob_sum = torch.zeros(shape, device=device)
    weight_sum = torch.zeros(shape, device=device)

    starts = [(x, y, z) for x in window_starts(shape[0], patch_size[0], stride[0])
              for y in window_starts(shape[1], patch_size[1], stride[1])
              for z in window_starts(shape[2], patch_size[2], stride[2])]
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        batch = torch.stack([image[(slice(None),) + tuple(slice(s, s + p) for s, p in zip(start, patch_size))]
                             for start in chunk]).to(device)
        probs = torch.sigmoid(model(batch))[:, 0]
        for start, prob in zip(chunk, probs):
            window = tuple(slice(s, s + p) for s, p in zip(start, patch_size))
            prob_sum[window] += prob * weights
            weight_sum[window] += weights
    return (prob_sum / weight_sum).cpu().numpy()


def remove_small_components(mask, min_voxels):
    components, n_components = label(mask, structure=np.ones((3, 3, 3)))
    if n_components == 0:
        return mask.astype(bool)
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[components]


def keep_largest_component(mask):
    components, n_components = label(mask, structure=np.ones((3, 3, 3)))
    if n_components == 0:
        return mask.astype(bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax()
