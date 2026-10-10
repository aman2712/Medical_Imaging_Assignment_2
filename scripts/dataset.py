from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.preprocessing import normalize_channels, crop_to_box, augment


def cache_case(row, data_root, cache_dir):
    image_path = cache_dir / f"{row.case_id}_image.npy"
    label_path = cache_dir / f"{row.case_id}_label.npy"
    if image_path.exists() and label_path.exists():
        return
    image = nib.load(data_root / row.image_relpath).get_fdata().astype(np.float32)
    label = nib.load(data_root / row.label_relpath).get_fdata()
    np.save(image_path, crop_to_box(normalize_channels(image)).astype(np.float16))
    np.save(label_path, crop_to_box((label > 0).astype(np.uint8)))


class PatchDataset(Dataset):
    def __init__(self, case_ids, cache_dir, patch_size=(64, 64, 64), sampling="uniform",
                 tumor_prob=0.5, samples_per_epoch=400, train=True, seed=42):
        self.case_ids = list(case_ids)
        self.cache_dir = Path(cache_dir)
        self.patch_size = np.array(patch_size)
        self.sampling = sampling
        self.tumor_prob = tumor_prob
        self.samples_per_epoch = samples_per_epoch
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.images = [np.load(self.cache_dir / f"{c}_image.npy", mmap_mode="r") for c in self.case_ids]
        self.labels = [np.load(self.cache_dir / f"{c}_label.npy") for c in self.case_ids]
        self.tumor_voxels = [np.argwhere(label) for label in self.labels]

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _patch_start(self, case, rng):
        shape = np.array(self.labels[case].shape)
        if self.sampling == "tumor_aware" and rng.random() < self.tumor_prob and len(self.tumor_voxels[case]) > 0:
            center = self.tumor_voxels[case][rng.integers(len(self.tumor_voxels[case]))]
            start = center - self.patch_size // 2
        else:
            start = rng.integers(0, shape - self.patch_size + 1)
        return np.clip(start, 0, shape - self.patch_size)

    def _sample(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        case = rng.integers(len(self.case_ids))
        start = self._patch_start(case, rng)
        window = tuple(slice(s, s + p) for s, p in zip(start, self.patch_size))
        return rng, case, window

    def contains_tumor(self, idx):
        _, case, window = self._sample(idx)
        return bool(self.labels[case][window].any())

    def __getitem__(self, idx):
        rng, case, window = self._sample(idx)
        image = np.array(self.images[case][window], dtype=np.float32)  # [X, Y, Z, C]
        label = self.labels[case][window]  # [X, Y, Z]
        if self.train:
            image, label = augment(image, label, rng)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(3, 0, 1, 2)))  # [C, D, H, W]
        label = torch.from_numpy(np.ascontiguousarray(label[None])).float()  # [1, D, H, W]
        return image, label
