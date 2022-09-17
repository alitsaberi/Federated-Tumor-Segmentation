from typing import Union, Sequence

from monai.networks.nets import SwinUNETR


def build_model(
    img_size: Union[Sequence[int], int],
    feature_size: int = 48,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = False,
):

    model = SwinUNETR(
        img_size=img_size,
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )

    return model
