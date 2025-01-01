# -- LIBRARY IMPORTS --

import yaml

from utils import RESOURCES_DATA, RESOURCES_WEIGHTS
from video_ssl.configs import video_ssl as exp_cfg

# -- CONFIGURATION --


def override_for_ucf101(n_frames: int, width: int, height: int):
    with open(
        "models/official/projects/video_ssl/configs/experiments/cvrl_linear_eval_k600.yaml",
        "r",
    ) as file:
        override_params = yaml.full_load(file)

    exp_config = exp_cfg.exp_factory.get_exp_config("video_ssl_linear_eval_kinetics600")
    exp_config.override(override_params, is_strict=False)

    # Runtime configuration.
    exp_config.runtime.distribution_strategy = "mirrored"

    # Task configuration.
    exp_config.task.freeze_backbone = True
    exp_config.task.init_checkpoint = (
        RESOURCES_WEIGHTS / "r3d_1x_k600_800ep" / "r3d_1x_k600_800ep_backbone-1"
    ).as_posix()
    exp_config.task.init_checkpoint_modules = "backbone"

    # Model configuration.
    exp_config.task.model.projection_dim = 10

    # Training data configuration.
    exp_config.task.train_data.input_path = (
        RESOURCES_DATA / "UCF101_subset" / "records" / "train*"
    ).as_posix()
    exp_config.task.train_data.num_classes = 10
    exp_config.task.train_data.global_batch_size = 2
    exp_config.task.train_data.min_image_size = width
    exp_config.task.train_data.num_examples = 400
    exp_config.task.train_data.feature_shape = (n_frames, height, width, 3)

    # Validation data configuration.
    exp_config.task.validation_data.num_classes = 10
    exp_config.task.validation_data.input_path = (
        RESOURCES_DATA / "UCF101_subset" / "records" / "val*"
    ).as_posix()
    exp_config.task.validation_data.global_batch_size = 2
    exp_config.task.validation_data.min_image_size = width
    exp_config.task.validation_data.num_examples = 100
    exp_config.task.validation_data.feature_shape = (n_frames, height, width, 3)

    # Trainer configuration.
    exp_config.trainer.train_steps = 2000
    exp_config.trainer.checkpoint_interval = 200
    exp_config.trainer.steps_per_loop = 200
    exp_config.trainer.summary_interval = 200
    exp_config.trainer.validation_interval = 200
    exp_config.trainer.validation_steps = 200
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = 2000
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = (
        0.008
    )
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.007
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 200

    return exp_config
