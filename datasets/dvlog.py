"""
This file contains the DVlog dataset class and dataloader.
"""

from pathlib import Path
from typing import Union
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class DVlog(data.Dataset):
    def __init__(self, root: Union[str, Path], fold: str = "train"):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.features = []
        self.labels = []

        with open(self.root / "labels.csv", "r") as f:
            for line in f:
                sample = line.strip().split(",")
                if self.is_sample(sample):
                    s_id = sample[0]
                    s_label = int(sample[1] == "depression")
                    self.labels.append(s_label)

                    v_feature_path = self.root / s_id / f"{s_id}_visual.npy"
                    a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"

                    v_feature = np.load(v_feature_path)  # [frames, num_v(136)]
                    a_feature = np.load(a_feature_path)  # [frames, num_a(25)]

                    # concat visual and acoustic features along the 2nd axis
                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    if T_v == T_a:
                        feature = np.concatenate((v_feature, a_feature), axis=1).astype(
                            np.float32
                        )  # [frames, 136+25 = 161]
                    else:
                        T = min(T_v, T_a)
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    self.features.append(feature)

        print(
            f"ALL:{len(self.labels)}, Positive:{np.sum(self.labels)}, Negative:{len(self.labels)-np.sum(self.labels)}"
        )

    def is_sample(self, sample) -> bool:
        fold = sample[4]
        return fold == self.fold

    def __getitem__(self, i: int):
        feature = self.features[i]
        label = self.labels[i]
        return feature, label

    def __len__(self):
        return len(self.labels)


def _collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True
    )
    padding_mask = (padded_features.sum(dim=-1) != 0).long()
    labels = torch.tensor(labels)
    return padded_features, labels, padding_mask


def get_dvlog_dataloader(
    root: Union[str, Path], fold: str = "train", batch_size: int = 16
):
    """
    Get the dataloader for DVlog dataset.

    Args:
        root: path to the dataset.
        fold: "train", "valid", or "test".
        batch_size: batch size. By default 16.

    Returns:
        dataloader: the dataloader for DVlog dataset.
    """
    dataset = DVlog(root, fold)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=(fold == "train"),
    )
    return dataloader
