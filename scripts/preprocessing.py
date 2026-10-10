import numpy as np

CROP_START = (36, 19, 0)
CROP_SIZE = (168, 192, 152)

def normalize_channels(image_data):
    normalized = np.zeros_like(image_data, dtype=np.float32)
    for c in range(image_data.shape[-1]):
        channel = image_data[..., c]
        brain = channel != 0
        values = channel[brain]
        normalized[..., c][brain] = (values - values.mean()) / (values.std() + 1e-8)
    return normalized

def crop_to_box(volume, start=CROP_START, size=CROP_SIZE):
    out = np.zeros(tuple(size) + volume.shape[3:], dtype=volume.dtype)
    src = [slice(max(s, 0), min(s + n, d)) for s, n, d in zip(start, size, volume.shape[:3])]
    dst = [slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start)]
    out[tuple(dst)] = volume[tuple(src)]
    return out


def augment(image, label, rng):
    image, label = image.copy(), label.copy()
    if rng.random() < 0.5:
        image, label = image[::-1], label[::-1]
    for c in range(image.shape[-1]):
        brain = image[..., c] != 0
        scale, shift = rng.uniform(0.9, 1.1), rng.uniform(-0.1, 0.1)
        noise = rng.normal(0, 0.05, brain.sum())
        image[..., c][brain] = image[..., c][brain] * scale + shift + noise
    return image, label
