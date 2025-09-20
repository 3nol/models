"""Video encoding configuration definition."""

from typing import List, Tuple
import dataclasses
from official import vision
from official.core import exp_factory
import official.core.config_definitions as cfg
import official.modeling.hyperparams.base_config as hyperparams
import official.vision.configs.backbones_3d as backbones
from official.vision.configs import video_classification

Losses = video_classification.Losses
VisionTransformer = vision.configs.backbones.VisionTransformer
VideoClassificationModel = video_classification.VideoClassificationModel
VideoClassificationTask = video_classification.VideoClassificationTask


@dataclasses.dataclass
class ViTDataConfig(video_classification.DataConfig):
    is_ssl: bool = False
    memory_bank_size: int = -1
    memory_bank_dim: int = -1

    # Whether to apply temporal pace augmentations.
    temporal_pace_modeling: bool = False


@dataclasses.dataclass
class ViTBackboneConfig(VisionTransformer):
    model_name: str = "tubevit-ti16"

    @dataclasses.dataclass
    class TubeBlock(hyperparams.Config):
        """Stores a single triple of T, H, W."""

        t: int = 1
        h: int = 1
        w: int = 1

        def as_tuple(self) -> Tuple[int, int, int]:
            return self.t, self.h, self.w

        @classmethod
        def from_list(
            cls, tuples: List[Tuple[int, int, int]]
        ) -> List["ViTBackboneConfig.TubeBlock"]:
            return [cls(t=tu[0], h=tu[1], w=tu[2]) for tu in tuples]

    # Instead of List[Tuple[int, int, int]], use a list of TubeBlock.
    kernel_sizes: List[TubeBlock] = dataclasses.field(
        default_factory=lambda: ViTBackboneConfig.TubeBlock.from_list(
            [(4, 8, 8), (8, 4, 4), (2, 12, 12), (1, 16, 16)]
        )
    )
    strides: List[TubeBlock] = dataclasses.field(
        default_factory=lambda: ViTBackboneConfig.TubeBlock.from_list(
            [(8, 32, 32), (3, 32, 32), (8, 32, 32), (16, 16, 16)]
        )
    )
    offsets: List[TubeBlock] = dataclasses.field(
        default_factory=lambda: ViTBackboneConfig.TubeBlock.from_list(
            [(0, 0, 0), (4, 8, 8), (0, 16, 16), (0, 0, 0)]
        )
    )


@dataclasses.dataclass
class ViTModelConfig(VideoClassificationModel):
    normalize_feature: bool = True

    @dataclasses.dataclass
    class ExtendedBackbone3D(backbones.Backbone3D):
        """Extended configuration for 3D backbones.

        Attributes:
            type: 'str', type of backbone be used, one of the fields below.
            resnet_3d: resnet3d backbone config.
            resnet_3d_rs: resnet3d-rs backbone config.
            tubevit: tubevit backbone config.
        """

        tubevit: ViTBackboneConfig = dataclasses.field(
            default_factory=ViTBackboneConfig
        )

    backbone: ExtendedBackbone3D = dataclasses.field(
        default_factory=lambda: ViTModelConfig.ExtendedBackbone3D(
            type="tubevit",
            resnet_3d=None,
            resnet_3d_rs=None,
            tubevit=ViTBackboneConfig(),
        )
    )


@dataclasses.dataclass
class ViTLossesConfig(Losses):
    label_smoothing = 0.1


@dataclasses.dataclass
class ViTPretrainTaskConfig(VideoClassificationTask):
    """Task config for ViT pretraining."""

    model: ViTModelConfig = dataclasses.field(default_factory=ViTModelConfig)
    train_data: ViTDataConfig = dataclasses.field(
        default_factory=lambda: ViTDataConfig(is_training=True, drop_remainder=True)
    )
    validation_data: ViTDataConfig = dataclasses.field(
        default_factory=lambda: ViTDataConfig(is_training=True, drop_remainder=False)
    )
    losses: ViTLossesConfig = dataclasses.field(default_factory=ViTLossesConfig)


@dataclasses.dataclass
class ViTLinearEvalTaskConfig(VideoClassificationTask):
    """Task config for ViT linear evaluation."""

    model: ViTModelConfig = dataclasses.field(default_factory=ViTModelConfig)
    train_data: ViTDataConfig = dataclasses.field(
        default_factory=lambda: ViTDataConfig(is_training=True, drop_remainder=True)
    )
    validation_data: ViTDataConfig = dataclasses.field(
        default_factory=lambda: ViTDataConfig(is_training=True, drop_remainder=False)
    )
    losses: ViTLossesConfig = dataclasses.field(default_factory=ViTLossesConfig)

    # Set the backbone to non-trainable (only the head is active).
    freeze_backbone: bool = True


# -- FACTORY-REGISTERED VIT EXPERIMENTS --


@exp_factory.register_config_factory("video_vit_pretrain_ucf101")
def video_vit_pretrain_ucf101() -> cfg.ExperimentConfig:
    """Pretraining of ViT Video classification on UCF101."""

    exp = video_classification.video_classification_ucf101()
    # TODO: Pre-train config is missing here.
    if exp is None:
        return None
    return exp


@exp_factory.register_config_factory("video_vit_linear_eval_ucf101")
def video_vit_linear_eval_ucf101() -> cfg.ExperimentConfig:
    """Linear evaluation of ViT Video classification on UCF101."""

    exp = video_classification.video_classification_ucf101()
    # TODO: Linear evaluation config is missing here.
    if exp is None:
        return None
    return exp
