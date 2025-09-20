"""Video vit pretrain task definition."""

from typing import Any, Optional, List, Tuple
import tensorflow as tf, tf_keras
from official import vision
from official.core import task_factory

from official.projects.tubevit.configs import tubevit as tubevit_cfg


@task_factory.register_task_cls(tubevit_cfg.ViTPretrainTaskConfig)
class VideoViTPretrainTask(vision.VideoClassificationTask):
    """A task for video vit pretraining."""

    def train_step(
        self,
        inputs: Tuple[Any, Any],
        model: tf_keras.Model,
        optimizer: tf_keras.optimizers.Optimizer,
        metrics: Optional[List[Any]] = None,
    ):
        """Does forward and backward.

        Args:
        inputs: a dictionary of input tensors.
        model: the model, forward pass definition.
        optimizer: the optimizer for this training step.
        metrics: a nested structure of metrics objects.

        Returns:
        A dictionary of logs.
        """

        features, labels = inputs

        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            # Casting output layer as float32 is necessary when mixed_precision is
            # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
            outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
            tf.summary.histogram(name="outputs", data=outputs)

            # Computes per-replica loss.
            if self._is_multilabel():
                outputs = tf.nest.map_structure(tf.math.sigmoid, outputs)
            else:
                outputs = tf.nest.map_structure(tf.math.softmax, outputs)

            all_losses = self.build_losses(
                model_outputs=outputs, labels=labels, aux_losses=model.losses
            )
            loss = all_losses[self.loss]
            scaled_loss = loss

            # For mixed_precision policy, when LossScaleOptimizer is used, loss is
            # scaled for numerical stability.
            if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)

        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)
        # Scales back gradient before apply_gradients when LossScaleOptimizer is
        # used.
        if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        grads_and_vars = []
        for g, v in zip(grads, tvars):
            name = str(v.name).replace(":", "_")
            tf.summary.histogram(name=f"gradients/{name}", data=g)
            tf.summary.histogram(name=f"weights/{name}", data=v)
            grads_and_vars.append((g, v))
        optimizer.apply_gradients(grads_and_vars)

        logs = all_losses
        if metrics:
            self.process_metrics(metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics})
        elif model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    # def validation_step(self, inputs, model, metrics=None):
    #     raise NotImplementedError

    # def inference_step(self, features, model):
    #     raise NotImplementedError
