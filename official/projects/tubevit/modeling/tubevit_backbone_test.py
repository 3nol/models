import tensorflow as tf, tf_keras

from official.projects.tubevit.configs import tubevit as tubevit_cfg
from official.projects.tubevit.modeling import tubevit_backbone


class TubeVisionTransformerTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.num_frames = 32
        self.height = 224
        self.width = 224
        self.batch_size = 2

        # Common backbone config.
        self.backbone_config = tubevit_cfg.ViTBackboneConfig()
        self.backbone_config.model_name = "tubevit-ti16"
        self.backbone_config.transformer.dropout_rate = 0.2
        self.backbone_config.representation_size = 64

        # Input specs.
        self.input_specs = {
            "image": tf_keras.layers.InputSpec(
                shape=[self.batch_size, self.num_frames, self.height, self.width, 3]
            )
        }

        # Built model.
        self.built_backbone = None

    @property
    def backbone(self):
        if self.built_backbone is None:
            self.built_backbone = tubevit_backbone.build_video_vit_backbone(
                input_specs=self.input_specs,
                backbone_config=tubevit_cfg.ViTModelConfig.ExtendedBackbone3D(
                    type="tubevit", tubevit=self.backbone_config
                ),
            )

        return self.built_backbone

    def test_model_creation(self):
        self.assertIsInstance(self.backbone, tf_keras.Model)

    def test_forward_pass(self):
        inputs = tf.ones([self.batch_size, self.num_frames, self.height, self.width, 3])
        outputs = self.backbone(inputs)["pre_logits"]

        # Check output shape.
        expected_shape = (self.batch_size, self.backbone_config.representation_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_forward_pass_with_different_tubes(self):
        model = tubevit_backbone.TubeVisionTransformer(
            input_specs={
                "image": tf_keras.layers.InputSpec(
                    shape=[self.batch_size, 16, self.height, self.width, 3]
                )
            },
            kernel_sizes=[(4, 8, 8), (8, 4, 4), (2, 12, 12), (1, 16, 16)],
            strides=[(8, 32, 32), (3, 32, 32), (8, 32, 32), (16, 16, 16)],
            offsets=[(0, 0, 0), (4, 8, 8), (0, 16, 16), (0, 0, 0)],
            representation_size=self.backbone_config.representation_size,
        )

        inputs = tf.ones([self.batch_size, 16, self.height, self.width, 3])
        outputs = model(inputs)["pre_logits"]

        # Check output shape.
        expected_shape = (self.batch_size, self.backbone_config.representation_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_serialize_deserialize(self):
        config = self.backbone.get_config()
        new_model = tubevit_backbone.TubeVisionTransformer.from_config(config)

        # Validate that the config can be used to create a new model.
        self.assertIsInstance(new_model, tf_keras.Model)


if __name__ == "__main__":
    tf.test.main()
