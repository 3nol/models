import functools
import os
import random
import logging

import orbit
import numpy as np
import tensorflow as tf
from official import vision
from official.core import exp_factory
from official.core import task_factory
from official.vision.dataloaders import tfexample_utils

from official.projects.tubevit.tasks import pretrain


class VideoClassificationTaskTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        data_dir = os.path.join(self.get_temp_dir(), "data")
        tf.io.gfile.makedirs(data_dir)
        self._data_path = os.path.join(data_dir, "data.tfrecord")
        # pylint: disable=g-complex-comprehension
        examples = [
            tfexample_utils.make_video_test_example(
                image_shape=(36, 36, 3),
                audio_shape=(20, 128),
                label=random.randint(0, 100),
            )
            for _ in range(2)
        ]
        # pylint: enable=g-complex-comprehension
        tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

    def test_task(self):
        config = exp_factory.get_exp_config("video_vit_pretrain_ucf101")
        config.task.train_data.global_batch_size = 2
        config.task.train_data.input_path = self._data_path
        config = vision.configs.video_classification.add_trainer(
            config,
            train_batch_size=config.task.train_data.global_batch_size,
            eval_batch_size=config.task.validation_data.global_batch_size,
            warmup_epochs=0,
        )

        task = pretrain.VideoViTPretrainTask(config.task)
        model = task.build_model()
        metrics = task.build_metrics()
        strategy = tf.distribute.get_strategy()

        dataset = orbit.utils.make_distributed_dataset(
            strategy, functools.partial(task.build_inputs), config.task.train_data
        )
        iterator = iter(dataset)
        optimizer = task.create_optimizer(config.trainer.optimizer_config)

        # Test only training step...
        initial_vars = [v.numpy().copy() for v in model.trainable_variables]
        for step in [
            lambda: task.train_step(next(iterator), model, optimizer, metrics=metrics),
        ] * 2:
            logs = step()
            # Assert that all losses (and metrics) were computed.
            self.assertIn("class_loss", logs)
            self.assertIn("reg_loss", logs)
            self.assertIn("loss", logs)
            self.assertIn("accuracy", logs)
            self.assertIn("top_1_accuracy", logs)
            self.assertIn("top_5_accuracy", logs)
            self.assertIn("variance", logs)
            self.assertIn("covariance", logs)
            self.assertIn("rankme", logs)
            # Assert set of trainable variables during training.
            self.assertLen(model.trainable_variables, 200)

        # ...but make sure every variable is trained.
        threshold = 1e-6
        for initial, trained in zip(initial_vars, model.trainable_variables):
            delta = np.max(np.abs(trained.numpy() - initial))
            if delta < threshold:
                logging.getLogger(__name__).warning(
                    f"small delta for '{trained.name}': {delta}"
                )
            self.assertGreater(delta, threshold**2)

    def test_task_factory(self):
        config = exp_factory.get_exp_config("video_vit_pretrain_ucf101")
        task = task_factory.get_task(config.task)
        self.assertIs(type(task), pretrain.VideoViTPretrainTask)


if __name__ == "__main__":
    tf.test.main()
