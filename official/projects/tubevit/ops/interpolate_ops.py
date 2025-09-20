import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec([None, None, None, None, None], tf.float32),
        tf.TensorSpec([3], tf.int32),
    ],
    jit_compile=True,
)
def interpolate_trilinear_5d(
    kernel_5d: tf.Tensor,
    size: tuple[int, int, int],
    name: str = "interpolate_trilinear_5d",
) -> tf.Tensor:
    """
    Specialized 3D "trilinear" resizing for a 5D filter in NDHWC-like format
    ```
      [D_in, H_in, W_in, C_in, C_out]
    ```
    to [D_out, H_out, W_out, C_in, C_out].

    Internally, we use the following steps.
      1) Merge the last two dims into a single "channels" axis => [D_in, H_in, W_in, C_in * C_out]
      2) Insert batch=1 => [1, D_in, H_in, W_in, C_in * C_out] for the progressive NDHWC resize.
      3) Resize depth, then height, then width via tf.image.resize (bilinear).
      4) Squeeze batch dim => [D_out, H_out, W_out, C_in * C_out]
      5) Split "C_in * C_out" back => [D_out, H_out, W_out, C_in, C_out]

    Note that we used assistance from LLM-based code generation to port this function from
    PyTorch's `torch.nn.functional.interpolate(..., ..., mode="trilinear")` to TensorFlow.

    Args:
      kernel_5d: A float Tensor of shape [D_in, H_in, W_in, C_in, C_out].
      size: A tuple (D_out, H_out, W_out).
      name: Operation name scope.

    Returns:
      A Tensor of shape [D_out, H_out, W_out, C_in, C_out].
    """
    with tf.name_scope(name):
        if len(kernel_5d.shape) != 5:
            raise ValueError(
                f"Expected 5D kernel [D, H, W, C_in, C_out], but got {kernel_5d.shape}"
            )

        # 1) Flatten C_in and C_out into single channels dimension.
        D_out, H_out, W_out = size[0], size[1], size[2]
        shape = tf.shape(kernel_5d)
        D_in, H_in, W_in, C_in, C_out = shape[0], shape[1], shape[2], shape[3], shape[4]

        # 2) Insert batch dimension => [1, D_in, H_in, W_in, C'] for NDHWC ops.
        flattened = tf.expand_dims(
            tf.reshape(kernel_5d, [D_in, H_in, W_in, -1]), axis=0
        )

        # 3) Progressive bilinear resize along D, then H, then W in NDHWC.
        def resize_along_axis(x, axis, out_size):
            forward_perm = [0, 1, 2, 3, 4]
            forward_perm[1], forward_perm[axis + 1] = (
                forward_perm[axis + 1],
                forward_perm[1],
            )
            transposed = tf.transpose(x, forward_perm)

            s = tf.shape(transposed)
            f = s[2] * s[3] * s[4]
            reshaped = tf.reshape(transposed, [s[0], s[1], f, 1])
            resized = tf.image.resize(reshaped, [out_size, f], method="bilinear")
            reshaped = tf.reshape(resized, [s[0], out_size, s[2], s[3], s[4]])

            backward_perm = [forward_perm.index(i) for i in range(5)]
            return tf.transpose(reshaped, backward_perm)

        # 4) Resize all three axes: depth, height, and width.
        merged = flattened
        merged = resize_along_axis(merged, 0, out_size=D_out)
        merged = resize_along_axis(merged, 1, out_size=H_out)
        merged = resize_along_axis(merged, 2, out_size=W_out)

        # 5) Remove batch dimension => [D_out, H_out, W_out, C_in * C_out].
        flattened = tf.squeeze(merged, axis=0)

        # 6) Split "C_in * C_out" => [D_out, H_out, W_out, C_in, C_out].
        kernel_5d = tf.reshape(merged, [D_out, H_out, W_out, C_in, C_out])
        return kernel_5d
