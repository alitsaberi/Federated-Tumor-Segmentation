from pathlib import Path
from typing import Union, Callable, Tuple

import pandas as pd

from monai import data, transforms

from utils.helpers import EnumBase


class FeTS2022(data.Dataset):
    class Partitioning(EnumBase):
        NATURAL = 1
        ARTIFICIAL = 2

        @property
        def partitioning_file_name(self):
            return f"partitioning_{self.value}.csv"

    def __init__(
        self,
        root_dir: Union[str, Path],
        train: bool = True,
        partitioning: str = "natural",
        partition_id: int = 0,
        transform: Callable = None,
        roi: Tuple[int] = (128, 128, 128),
    ):

        try:
            self.partitioning = FeTS2022.Partitioning.from_str(partitioning)
        except KeyError:
            raise ValueError(
                f"The partitioning {partitioning} is not valid. "
                f"Partitioning can have one of these values: "
                f"{FeTS2022.Partitioning.all()}"
            )

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        split_dir_name = (
            "MICCAI_FeTS2022_TrainingData"
            if train
            else "MICCAI_FeTS2022_ValidationData"
        )
        self.root_dir = root_dir / split_dir_name
        self.partition_id = partition_id
        self.data = self._get_data()

        self.transform = transform
        if self.transform is None:
            self.transform = (
                transforms.Compose(
                    [
                        transforms.LoadImaged(keys=["image", "label"]),
                        transforms.ConvertToMultiChannelBasedOnBratsClassesd(
                            keys="label"
                        ),
                        transforms.CropForegroundd(
                            keys=["image", "label"],
                            source_key="image",
                            k_divisible=[roi[0], roi[1], roi[2]],
                        ),
                        transforms.RandSpatialCropd(
                            keys=["image", "label"],
                            roi_size=[roi[0], roi[1], roi[2]],
                            random_size=False,
                        ),
                        transforms.RandFlipd(
                            keys=["image", "label"], prob=0.5, spatial_axis=0
                        ),
                        transforms.RandFlipd(
                            keys=["image", "label"], prob=0.5, spatial_axis=1
                        ),
                        transforms.RandFlipd(
                            keys=["image", "label"], prob=0.5, spatial_axis=2
                        ),
                        transforms.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                        transforms.RandScaleIntensityd(
                            keys="image", factors=0.1, prob=1.0
                        ),
                        transforms.RandShiftIntensityd(
                            keys="image", offsets=0.1, prob=1.0
                        ),
                    ]
                )
                if train
                else transforms.Compose(
                    [
                        transforms.LoadImaged(keys=["image", "label"]),
                        transforms.ConvertToMultiChannelBasedOnBratsClassesd(
                            keys="label"
                        ),
                        transforms.NormalizeIntensityd(
                            keys="image", nonzero=True, channel_wise=True
                        ),
                    ]
                )
            )

    def _prepare_data(self, subject):
        subject_id = subject["Subject_ID"]
        subject_dir = self.root_dir / subject_id

        d = {
            "image": [
                subject_dir / f"{subject_id}_flair.nii.gz",
                subject_dir / f"{subject_id}_t1ce.nii.gz",
                subject_dir / f"{subject_id}_t1.nii.gz",
                subject_dir / f"{subject_id}_t2.nii.gz",
            ],
            "label": subject_dir / f"{subject_id}_seg.nii.gz",
            "partition_id": subject["Partition_ID"],
        }

        return d

    def _get_data(self):
        partitioning_file_path = (
            self.root_dir / self.partitioning.partitioning_file_name
        )

        subjects_df = pd.read_csv(partitioning_file_path)
        if self.partition_id:
            if self.partition_id not in subjects_df["Partition_ID"]:
                raise ValueError(
                    f"The provided partition ID is not valid. "
                    f"Possible partition IDs: "
                    f"{subjects_df['Partition_ID'].unique()}"
                )

            subjects_df = subjects_df[
                subjects_df["Partition_ID"] == self.partition_id
            ]

        subjects_df = subjects_df.apply(
            self._prepare_data, axis=1, result_type="expand"
        )

        return subjects_df.to_dict("records")
