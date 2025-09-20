import unittest
import tensorflow as tf
import numpy as np

from official.projects.tubevit.ops.interpolate_ops import interpolate_trilinear_5d


class DummyInterpolator(tf.test.TestCase):
    """Example of 5D-interpolating test case."""

    def test_interpolation(self):
        # 1. Identity test.
        input_ = tf.constant(np.random.rand(4, 4, 4, 3, 5), dtype=tf.float32)
        result = interpolate_trilinear_5d(input_, size=tf.constant([4, 4, 4]))
        self.assertAllClose(result, input_)

        # 2. Simple up-sample by 2 test
        input_ = tf.constant(np.random.rand(2, 2, 2, 1, 1), dtype=tf.float32)
        result = interpolate_trilinear_5d(input_, size=tf.constant([4, 4, 4]))
        self.assertEqual(result.shape, (4, 4, 4, 1, 1))

        # 3. Simple down-sample by 2 test.
        input_ = tf.constant(np.random.rand(4, 4, 4, 2, 3), dtype=tf.float32)
        result = interpolate_trilinear_5d(input_, size=tf.constant([2, 2, 1]))
        self.assertEqual(result.shape, (2, 2, 1, 2, 3))

        # 4. Non-uniform test.
        input_ = tf.constant(np.random.rand(5, 6, 1, 3, 2), dtype=tf.float32)
        result = interpolate_trilinear_5d(input_, size=tf.constant([3, 2, 4]))
        self.assertEqual(result.shape, (3, 2, 4, 3, 2))

        # 5. One-dim mismatch.
        input_ = tf.constant(np.random.rand(2, 3, 4, 2, 2), dtype=tf.float32)
        result = interpolate_trilinear_5d(input_, size=tf.constant([2, 6, 4]))
        self.assertEqual(result.shape, (2, 6, 4, 2, 2))


if __name__ == "__main__":
    tf.test.main()
