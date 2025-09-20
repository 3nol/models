"""
Inspired by positional_encoding in "https://github.com/facebookresearch/pytorchvideo/blob/f7e7a88a9a04b70cb65a564acfc38538fe71ff7b/pytorchvideo/layers/positional_encoding.py".
Converted from PyTorch-based implementation in "https://github.com/daniel-code/TubeViT/blob/main/tubevit/positional_encoding.py" to TensorFlow.
"""

from typing import Tuple
import tensorflow as tf


def get_3d_sincos_pos_embed(
    embed_dim: int,
    tube_shape: Tuple[int, int, int],
    stride,
    offset,
    kernel_size,
    cls_token: bool = False,
) -> tf.Tensor:
    """
    Get 3D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension
        tube_shape: (t_size, grid_h_size, grid_w_size)
        kernel_size: Kernel size for adjusting grid positions
        offset: Offset values for adjusting grid positions
        stride: Stride values for adjusting grid positions
        cls_token: Whether to add classification token position

    Returns:
        tf.Tensor: [t_size*grid_size*grid_size, embed_dim] or
                  [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 3 * 2
    embed_dim_temporal = embed_dim // 3

    # Spatial encoding.
    grid_h_size = tube_shape[1]
    grid_h = tf.range(grid_h_size, dtype=tf.float32)
    grid_h = grid_h * stride[1] + offset[1] + kernel_size[1] // 2

    grid_w_size = tube_shape[2]
    grid_w = tf.range(tube_shape[2], dtype=tf.float32)
    grid_w = grid_w * stride[2] + offset[2] + kernel_size[2] // 2
    grid_w, grid_h = tf.meshgrid(grid_w, grid_h, indexing="ij")
    grid = tf.stack([grid_w, grid_h], axis=0)

    grid = tf.reshape(grid, [2, 1, grid_h_size, grid_w_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # Temporal encoding.
    t_size = tube_shape[0]
    grid_t = tf.range(t_size, dtype=tf.float32)
    grid_t = grid_t * stride[0] + offset[0] + kernel_size[0] // 2
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # Combine temporal and spatial embeddings
    pos_embed_temporal = pos_embed_temporal[:, None, :]
    pos_embed_temporal = tf.repeat(
        pos_embed_temporal, grid_h_size * grid_w_size, axis=1
    )
    pos_embed_spatial = pos_embed_spatial[None, :, :]
    pos_embed_spatial = tf.repeat(pos_embed_spatial, t_size, axis=0)

    pos_embed = tf.concat([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = tf.reshape(pos_embed, [-1, embed_dim])

    if cls_token:
        pos_embed = tf.concat([tf.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> tf.Tensor:
    """
    Get 2D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension
        grid_size: Grid height and width
        cls_token: Whether to add classification token position

    Returns:
        tf.Tensor: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = tf.range(grid_size, dtype=tf.float32)
    grid_w = tf.range(grid_size, dtype=tf.float32)
    grid_w, grid_h = tf.meshgrid(grid_w, grid_h, indexing="ij")
    grid = tf.stack([grid_w, grid_h], axis=0)

    grid = tf.reshape(grid, [2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = tf.concat([tf.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: tf.Tensor) -> tf.Tensor:
    """
    Get 2D sine-cosine positional embedding from grid.

    Args:
        embed_dim: Embedding dimension
        grid: Position coordinates tensor

    Returns:
        tf.Tensor: Position embeddings
    """
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = tf.concat([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: tf.Tensor) -> tf.Tensor:
    """
    Get 1D sine-cosine positional embedding.

    Args:
        embed_dim: Output dimension for each position
        pos: Positions to be encoded, size (M,)

    Returns:
        tf.Tensor: Encoded positions tensor of shape (M, D)
    """
    assert embed_dim % 2 == 0
    omega = tf.range(embed_dim // 2, dtype=tf.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / tf.pow(10000.0, omega)

    pos = tf.reshape(pos, [-1])
    out = tf.einsum("m,d->md", pos, omega)

    emb_sin = tf.sin(out)
    emb_cos = tf.cos(out)

    emb = tf.concat([emb_sin, emb_cos], axis=1)
    return emb
