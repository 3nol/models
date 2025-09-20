import functools
import os
import random

import orbit
import tensorflow as tf
from official.core import exp_factory
from official.core import task_factory
from official.vision.dataloaders import tfexample_utils

from official.projects.tubevit.tasks import linear_eval


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
                audio_shape=(20, 129),
                label=random.randint(0, 100),
            )
            for _ in range(2)
        ]
        # pylint: enable=g-complex-comprehension
        tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

    def test_task(self):
        config = exp_factory.get_exp_config("video_vit_linear_eval_ucf101")
        config.task.train_data.global_batch_size = 2
        config.task.train_data.input_path = self._data_path

        task = linear_eval.VideoViTEvalTask(config.task)
        model = task.build_model()
        metrics = task.build_metrics()
        strategy = tf.distribute.get_strategy()

        dataset = orbit.utils.make_distributed_dataset(
            strategy, functools.partial(task.build_inputs), config.task.train_data
        )
        iterator = iter(dataset)
        optimizer = task.create_optimizer(config.trainer.optimizer_config)

        # Test training and validation step.
        for step in [
            lambda: task.train_step(next(iterator), model, optimizer, metrics=metrics),
            lambda: task.validation_step(next(iterator), model, metrics=metrics),
        ]:
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
            # Assert set of trainable variables during validation.
            variables = model.trainable_variables
            self.assertLen(variables, 2)
            for variable in variables:
                self.assertStartsWith(variable.name, "dense")
                self.assertEqual(variable.shape[-1], config.task.train_data.num_classes)

    def test_task_factory(self):
        config = exp_factory.get_exp_config("video_vit_linear_eval_ucf101")
        task = task_factory.get_task(config.task)
        self.assertIs(type(task), linear_eval.VideoViTEvalTask)


if __name__ == "__main__":
    tf.test.main()
