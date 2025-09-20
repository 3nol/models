import tensorflow as tf
import numpy as np
import pathlib

from official.projects.tubevit.ops.posenc_ops import get_3d_sincos_pos_embed


class SinCosPosEmbedTest(tf.test.TestCase):

    def test_sincos_pos_embed(self):
        kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )

        tube_shape = (
            (2, 7, 7),
            (3, 7, 7),
            (2, 7, 7),
            (1, 14, 14),
        )

        actual_pos_encode = [tf.zeros(shape=(1, 768))]
        for i in range(len(kernel_sizes)):
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=768,
                tube_shape=tube_shape[i],
                stride=strides[i],
                offset=offsets[i],
                kernel_size=kernel_sizes[i],
            )
            actual_pos_encode.append(pos_embed)
        actual_pos_encode = tf.concat(actual_pos_encode, axis=0).numpy()
        expected_pos_encode = np.load( \
            (pathlib.Path(__file__) / ".." / "pos_encode.npy").as_posix())

        self.assertEqual(actual_pos_encode.shape, expected_pos_encode.shape)
        self.assertAllClose(
            actual_pos_encode, expected_pos_encode, rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    tf.test.main()
