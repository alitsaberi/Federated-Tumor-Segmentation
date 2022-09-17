from pathlib import Path
from typing import Union, Callable, Sequence, List, Optional

import pandas as pd

from monai import data, transforms

from utils.helpers import EnumBase


class Partitioning(EnumBase):
    NATURAL = 1
    ARTIFICIAL = 2

    @classmethod
    def from_str(
        cls, string: str, root_dir: Union[str, Path], train: bool = True
    ):
        obj = cls[string.upper()]

        split_dir_name = (
            "MICCAI_FeTS2022_TrainingData"
            if train
            else "MICCAI_FeTS2022_ValidationData"
        )

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        obj._root_dir = root_dir / split_dir_name
        obj._subjects_df = pd.read_csv(
            obj._root_dir / obj.partitioning_file_name
        )

        obj._train = train
        return obj

    @property
    def train(self) -> bool:
        return self._train

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def partitioning_file_name(self) -> str:
        return f"partitioning_{self.value}.csv"

    @property
    def subjects_df(self) -> pd.DataFrame:
        return self._subjects_df

    @property
    def partitions(self) -> List[int]:
        return self.subjects_df["Partition_ID"].unique().tolist()

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

    def get_data(self, partition_id: int = 0):

        subjects_df = self.subjects_df
        if partition_id:
            subjects_df = subjects_df[
                subjects_df["Partition_ID"] == partition_id
            ]

        subjects_df = subjects_df.apply(
            self._prepare_data, axis=1, result_type="expand"
        )

        return subjects_df.to_dict("records")


class FeTS2022(data.Dataset):
    def __init__(
        self,
        data: Sequence,
        train: bool = True,
        transform: Optional[Callable] = None,
        roi: Sequence[int] = (128, 128, 128),
    ):

        self.data = data

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
