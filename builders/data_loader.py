from pathlib import Path
from typing import Union, Sequence, Optional

from monai.data import DataLoader

from datasets.fets2022 import FeTS2022, Partitioning


def build_data_loaders(
    data_dir: Union[str, Path],
    roi: Union[Sequence[int], int],
    batch_size: int,
    partitioning: str = "natural",
    selected_partitions: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    num_workers: int = 8,
):

    partitioning = Partitioning.from_str(partitioning, data_dir)
    partitions = (
        selected_partitions if selected_partitions else partitioning.partitions
    )

    train_loaders = []
    val_loaders = []

    for partition_id in partitions:
        train_dataset = FeTS2022(
            partitioning.get_data(partition_id=partition_id),
            train=True,
            roi=roi,
        )
        train_loaders.append(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )
        )

    return train_loaders, val_loaders
