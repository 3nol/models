"""Build video embedding models."""

from typing import List, Tuple, Optional, Union, Any, Dict
import logging
import numpy as np
import tensorflow as tf, tf_keras
from official import vision
import official.modeling.hyperparams.base_config as hyperparams

from official.projects.tubevit.configs import tubevit as tubevit_cfg
from official.projects.tubevit.ops.posenc_ops import get_3d_sincos_pos_embed
from official.projects.tubevit.ops.interpolate_ops import interpolate_trilinear_5d
from official.vision.modeling.backbones.vit import TokenLayer, Encoder
from official.vision.modeling.backbones.vit_specs import VIT_SPECS

layers = tf_keras.layers


@tf_keras.utils.register_keras_serializable(package="Vision")
class SparseTubeTokenizer(layers.Layer):
    """
    A TensorFlow analog of the PyTorch `SparseTubesTokenizer`.
    This implementation creates a distinct trainable 3D convolution kernel
    + bias for each 'tube path', and applies them to the input video tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_sizes: List[Tuple[int, int, int]],
        strides: List[Tuple[int, int, int]],
        offsets: List[Tuple[int, int, int]],
        padding: str = "VALID",
        kernel_regularizer: Optional[
            Union[str, tf_keras.regularizers.Regularizer]
        ] = None,
        kernel_initializer: str = "lecun_normal",
        **kwargs,
    ):
        super(SparseTubeTokenizer, self).__init__(**kwargs)

        self._hidden_size = hidden_size
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._offsets = offsets
        self._padding = padding
        self._kernel_regularizer = kernel_regularizer
        self._kernel_initializer = kernel_initializer

        # Create a single kernel weight.
        self._conv_proj_weight = self.add_weight(
            name="tube_weight",
            shape=(*self._kernel_sizes[0], 3, self._hidden_size),
            regularizer=self._kernel_regularizer,
            initializer=self._kernel_initializer,
            trainable=True,
        )
        self._conv_proj_bias = self.add_weight(
            name="tube_bias",
            shape=(len(self._kernel_sizes), self._hidden_size),
            initializer="zeros",
            trainable=True,
        )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        base_config = super(SparseTubeTokenizer, self).get_config()
        config = dict(list(base_config.items()))
        config.update(
            {
                "hidden_size": self._hidden_size,
                "kernel_sizes": self._kernel_sizes,
                "strides": self._strides,
                "offsets": self._offsets,
                "padding": self._padding,
                "kernel_regularizer": self._kernel_regularizer,
                "kernel_initializer": self._kernel_initializer,
            }
        )
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
          inputs: A float tensor of shape [batch, T, H, W, 3].

        Returns:
          A float tensor of shape [batch, sum_of_tube_tokens, hidden_size],
          where sum_of_tube_tokens is the sum of (D' * H' * W') across all tubes.
        """
        tubes = []
        for i, (kernel_size, stride, offset) in enumerate(
            zip(self._kernel_sizes, self._strides, self._offsets)
        ):
            if i == 0:
                kernel = self._conv_proj_weight
            else:
                kernel = interpolate_trilinear_5d(
                    self._conv_proj_weight, size=kernel_size
                )

            # Slice out the region after offset.
            x = inputs[:, offset[0] :, offset[1] :, offset[2] :, :]

            # Perform 3D convolution with the given kernel.
            tube = tf.nn.bias_add(
                tf.nn.conv3d(
                    x,
                    filters=kernel,
                    strides=[1, *stride, 1],
                    padding=self._padding,
                    data_format="NDHWC",
                ),
                bias=self._conv_proj_bias[i],
                data_format="NDHWC",
            )

            # Flatten out D', H', W' into a single tokens dimension.
            s = tf.shape(tube)
            tubes.append(tf.reshape(tube, (s[0], s[1] * s[2] * s[3], s[4])))

        # Concat all tubes along the token dimension.
        return tf.concat(tubes, axis=1)


@tf_keras.utils.register_keras_serializable(package="Vision")
class SparseTubePositionEncoder(layers.Layer):
    """
    A layer that creates and adds a non-trainable (sine-cosine) 3D "tube"
    positional embedding to the input sequence of tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        video_shape: Tuple[int, int, int, int],
        kernel_sizes: List[Tuple[int, int, int]],
        strides: List[Tuple[int, int, int]],
        offsets: List[Tuple[int, int, int]],
        **kwargs,
    ):
        super(SparseTubePositionEncoder, self).__init__(**kwargs)

        self._hidden_size = hidden_size
        self._video_shape = video_shape
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._offsets = offsets

        self._pos_encoding = tf.Variable(
            name="pos_embedding",
            initial_value=self._get_pos_encoding(),
            trainable=False,
        )

    def get_config(self) -> Dict[str, Union[str, bool, list, tuple]]:
        """Returns a dictionary for recreating this layer."""
        base_config = super(SparseTubePositionEncoder, self).get_config()
        config = dict(list(base_config.items()))
        config.update(
            {
                "hidden_size": self._hidden_size,
                "kernel_sizes": self._kernel_sizes,
                "strides": self._strides,
                "offsets": self._offsets,
            }
        )
        return config

    def _tube_positional_encoding(
        self,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        offset: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Compute the output shape for a single tube path, e.g.:
           out_len = floor((in_len - offset - kernel_size) / stride) + 1
        for each of (time, height, width).
        """
        video_shape = np.array(self._video_shape, dtype=np.int32)
        kernel_size = np.array(kernel_size, dtype=np.int32)
        stride = np.array(stride, dtype=np.int32)
        offset = np.array(offset, dtype=np.int32)

        # The `self._video_shape`` is [D, H, W, C], so we want (D, H, W).
        in_dims = video_shape[[0, 1, 2]]
        out_shape = np.floor(((in_dims - offset - kernel_size) / stride) + 1)
        return out_shape.astype(int)

    def _get_pos_encoding(self) -> tf.Tensor:
        """
        Recreates the PyTorch _generate_position_embedding logic:
          1) Adds a leading zero vector for a [CLS] token.
          2) For each tube path, compute a 3D sine-cosine embedding of shape
             (D'*H'*W', hidden_dim).
          3) Concatenate them (plus the 1 for [CLS]) -> final shape is
             [1 + sum(D'*H'*W'), hidden_dim].
        """
        # Leading row for e.g. [CLS].
        all_embeddings = []
        cls_embed = tf.zeros((1, self._hidden_size), dtype=tf.float32)
        all_embeddings.append(cls_embed)

        for ksz, std, off in zip(self._kernel_sizes, self._strides, self._offsets):
            tube_shape = self._tube_positional_encoding(ksz, std, off)
            pos_embed_3d = get_3d_sincos_pos_embed(
                embed_dim=self._hidden_size,
                tube_shape=tube_shape,
                kernel_size=ksz,
                stride=std,
                offset=off,
                cls_token=False,
            )
            all_embeddings.append(pos_embed_3d)

        return tf.concat(all_embeddings, axis=0)

    def call(
        self,
        inputs: tf.Tensor,
        states: Optional[Dict[str, tf.Tensor]] = None,
        output_states: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict[str, tf.Tensor]]]:
        """
        Adds the precomputed (or on-the-fly) positional embedding to `inputs`.

        Args:
          inputs: [batch_size, num_tokens, hidden_dim]
          states: (Optional) dict for streaming or other stateful logic.
          output_states: If True, returns (outputs, states). Else returns outputs.

        Returns:
          outputs or (outputs, states), where outputs is
          [batch_size, num_tokens, hidden_dim].
        """
        states = dict(states) if states else {}

        pos_encoding = tf.cast(self._pos_encoding, dtype=inputs.dtype)
        # Expand to shape [1, num_tokens, hidden_dim].
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        # Add to inputs.
        outputs = inputs + pos_encoding
        return (outputs, states) if output_states else outputs


@tf_keras.utils.register_keras_serializable(package="Vision")
class SelfAttentionPooling1D(layers.Layer):
    """
    A Keras/TensorFlow implementation of Self-Attention Pooling:
      'Self-Attention Encoding and Pooling for Speaker Recognition'
      (https://arxiv.org/pdf/2008.01077v1.pdf)

    This layer learns to create a set of attention weights over the sequence
    dimension and forms a weighted average of the features.
    """

    def __init__(self, keepdims: bool = False, **kwargs):
        """
        Args:
          keepdims: If True, retains the time dimension of size 1 in the result.
        """
        super(SelfAttentionPooling1D, self).__init__(**kwargs)

        self._keepdims = keepdims
        # A Dense layer to produce scalar attention weights (one per time-step).
        self._weights = layers.Dense(1, use_bias=False)

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        base_config = super(SelfAttentionPooling1D, self).get_config()
        config = dict(list(base_config.items()))
        config.update(self._weights.get_config())
        config.update({"keepdims": self._keepdims})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Args:
          inputs: [batch_size, time_steps, hidden_size]

        Returns:
          If keepdims=False, shape is [batch_size, hidden_size].
          If keepdims=True, shape is [batch_size, 1, hidden_size].
        """
        # Compute attention weights over time dimension -> shape [B, T, 1].
        attention_weights = tf.nn.softmax(self._weights(inputs), axis=1)

        # Weighted sum of inputs along the time dimension -> shape [B, 1, H] or [B, H].
        return tf.reduce_sum(
            inputs * attention_weights, axis=1, keepdims=self._keepdims
        )


class TubeVisionTransformer(tf_keras.Model):
    """Class to build TubeVisionTransformer family model."""

    def __init__(
        self,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        init_stochastic_depth_rate=0.0,
        input_specs={"image": layers.InputSpec(shape=[None, 32, 224, 224, 3])},
        # patch_size=16,
        hidden_size=768,
        representation_size=0,
        kernel_sizes=[(8, 8, 8), (16, 4, 4), (4, 12, 12), (1, 16, 16)],
        strides=[(16, 32, 32), (6, 32, 32), (16, 32, 32), (32, 16, 16)],
        offsets=[(0, 0, 0), (4, 8, 8), (0, 16, 16), (0, 0, 0)],
        # pooler="token",
        kernel_regularizer: str = None,
        original_init: bool = True,
        # output_encoded_tokens: bool = True,
        # output_2d_feature_maps: bool = False,
        # pos_embed_shape: Optional[Tuple[int, int]] = None,
        layer_scale_init_value: float = 0.0,
        transformer_partition_dims: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ):
        """VisionTransformer initialization function."""
        inputs = {
            k: tf_keras.Input(name=k, shape=v.shape[1:]) for k, v in input_specs.items()
        }
        video = inputs["image"]
        self._config_dict = {
            "mlp_dim": mlp_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_dropout_rate": attention_dropout_rate,
            "dropout_rate": dropout_rate,
            "init_stochastic_depth_rate": init_stochastic_depth_rate,
            "input_specs": input_specs,
            "hidden_size": hidden_size,
            "representation_size": representation_size,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "offsets": offsets,
            "kernel_regularizer": kernel_regularizer,
            "original_init": original_init,
            "layer_scale_init_value": layer_scale_init_value,
            "transformer_partition_dims": transformer_partition_dims,
        }

        kernel_sizes = TubeVisionTransformer.build_tube_tuples(kernel_sizes)
        strides = TubeVisionTransformer.build_tube_tuples(strides)
        offsets = TubeVisionTransformer.build_tube_tuples(offsets)

        x = SparseTubeTokenizer(
            hidden_size=hidden_size,
            kernel_sizes=kernel_sizes,
            strides=strides,
            offsets=offsets,
            padding="VALID",
            kernel_regularizer=kernel_regularizer,
            kernel_initializer="lecun_normal" if original_init else "he_uniform",
        )(video)

        # We want to add a class token, add it here.
        x = TokenLayer(name="cls_token")(x)

        # Add position encoding for sparse tubes.
        x = SparseTubePositionEncoder(
            hidden_size=hidden_size,
            video_shape=video.shape[1:],
            kernel_sizes=kernel_sizes,
            strides=strides,
            offsets=offsets,
        )(x)

        # Use in-built transformer encoder.
        x = Encoder(
            name="transformer_encoder",
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=(
                dict(
                    class_name="TruncatedNormal",
                    config=dict(stddev=0.02),
                )
                if original_init
                else "glorot_uniform"
            ),
            init_stochastic_depth_rate=init_stochastic_depth_rate,
            add_pos_embed=False,
            layer_scale_init_value=layer_scale_init_value,
            transformer_partition_dims=transformer_partition_dims,
        )(x)

        # Apply self-attention pooling to get a single representation.
        x = SelfAttentionPooling1D(name="attention_pooling")(x)

        if representation_size:
            x = layers.Dense(
                name="pre_logits",
                units=representation_size,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer="lecun_normal" if original_init else "he_uniform",
                activation="tanh",
            )(x)
        else:
            x = layers.Identity(name="pre_logits")(x)
        x = layers.Reshape(
            name="shape_output",
            target_shape=(representation_size or hidden_size,),
        )(x)

        super().__init__(inputs=inputs, outputs={"pre_logits": x}, **kwargs)

    @classmethod
    def build_tube_tuples(cls, sizes: List[Any]) -> List[Tuple[int, int, int]]:
        if len(sizes) == 0:
            raise ValueError("Tube sizes must not be empty")
        if isinstance(sizes[0], tubevit_cfg.ViTBackboneConfig.TubeBlock):
            return [size.as_tuple() for size in sizes]
        if isinstance(sizes[0], tuple):
            return sizes
        raise ValueError("Tube tuples must be tuples themselves or a TubeBlock")

    @property
    def checkpoint_items(self):
        """Returns a dictionary of items to be additionally checkpointed."""
        return dict()

    @property
    def backbone(self):
        return self

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


@vision.backbone_factory.register_backbone_builder("tubevit")
def build_video_vit_backbone(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config = None,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
):
    """Build TubeViT backbone."""
    del norm_activation_config
    backbone_type = backbone_config.type
    if backbone_type != "tubevit":
        raise ValueError(f"Inconsistent backbone type {backbone_type}")

    backbone_cfg = backbone_config.get()
    backbone_name = backbone_cfg.model_name.split("tube", 1)[1]
    if backbone_name in VIT_SPECS:
        backbone_cfg.override(VIT_SPECS[backbone_name])
    logging.getLogger(__name__).info(
        (
            "TubeViT specs: mlp_dim=%d, num_heads=%d, num_layers=%d, "
            "hidden_size=%d, representation_size=%d"
        ),
        backbone_cfg.transformer.mlp_dim,
        backbone_cfg.transformer.num_heads,
        backbone_cfg.transformer.num_layers,
        backbone_cfg.hidden_size,
        backbone_cfg.representation_size,
    )
    if not isinstance(input_specs, dict):
        input_specs = {"image": input_specs}

    model = TubeVisionTransformer(
        mlp_dim=backbone_cfg.transformer.mlp_dim,
        num_heads=backbone_cfg.transformer.num_heads,
        num_layers=backbone_cfg.transformer.num_layers,
        attention_dropout_rate=backbone_cfg.transformer.attention_dropout_rate,
        dropout_rate=backbone_cfg.transformer.dropout_rate,
        init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
        input_specs=input_specs,
        hidden_size=backbone_cfg.hidden_size,
        representation_size=backbone_cfg.representation_size,
        kernel_sizes=backbone_cfg.kernel_sizes,
        strides=backbone_cfg.strides,
        offsets=backbone_cfg.offsets,
        kernel_regularizer=l2_regularizer,
        original_init=backbone_cfg.original_init,
        layer_scale_init_value=backbone_cfg.layer_scale_init_value,
        transformer_partition_dims=backbone_cfg.transformer_partition_dims,
    )

    return model
