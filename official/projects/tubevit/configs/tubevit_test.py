from absl.testing import parameterized
import tensorflow as tf
import official.core.config_definitions as cfg
from official.core import exp_factory

import official.projects.tubevit.configs.tubevit as tubevit_cfg


class VideoClassificationConfigTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.parameters(
        ("video_vit_pretrain_ucf101",),
    )
    def test_video_vit_pretrain_configs(self, config_name):
        config = exp_factory.get_exp_config(config_name)
        self.assertIsInstance(config, cfg.ExperimentConfig)
        self.assertIsInstance(config.task, tubevit_cfg.ViTPretrainTaskConfig)
        self.assertIsInstance(config.task.model, tubevit_cfg.ViTModelConfig)
        self.assertIsInstance(config.task.losses, tubevit_cfg.ViTLossesConfig)
        self.assertIsInstance(config.task.train_data, tubevit_cfg.ViTDataConfig)

        config.task.train_data.is_training = None
        with self.assertRaises(KeyError):
            config.validate()

    @parameterized.parameters(
        ("video_vit_linear_eval_ucf101",),
    )
    def test_video_vit_linear_eval_configs(self, config_name):
        config = exp_factory.get_exp_config(config_name)
        self.assertIsInstance(config, cfg.ExperimentConfig)
        self.assertIsInstance(config.task, tubevit_cfg.ViTLinearEvalTaskConfig)
        self.assertIsInstance(config.task.model, tubevit_cfg.ViTModelConfig)
        self.assertIsInstance(config.task.train_data, tubevit_cfg.ViTDataConfig)

        config.task.train_data.is_training = None
        with self.assertRaises(KeyError):
            config.validate()


if __name__ == "__main__":
    tf.test.main()
