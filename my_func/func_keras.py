import tensorflow as tf
import numpy as np

def pre_encoder_keras(I, dims=37):
    I = tf.math.floormod(I, 2.0 * np.pi)
    I = tf.where(I > np.pi,
                 I - 2.0 * np.pi,
                 I)

    interval = tf.cast(dims - 1, tf.float32)
    I_encoded = (I + np.pi) / np.pi * interval / 2.0

    I_integer   = tf.cast(tf.floor(I_encoded), tf.int64)
    I_frac      = tf.expand_dims(I_encoded - tf.cast(I_integer, tf.float32), -1)

    one_hot_int   = tf.one_hot(I_integer, depth=dims, dtype=tf.float32)
    one_hot_next  = tf.one_hot((I_integer + 1) % dims, depth=dims, dtype=tf.float32)

    I_encoded = (1.0 - I_frac) * one_hot_int + I_frac * one_hot_next

    return I_encoded

def cann3d_pre_encoder_keras(I: tf.Tensor,  dims: int = 37, concat: bool = False) -> tf.Tensor:
    # 取出三列
    I_x = tf.expand_dims(I[:, 0], -1)  # (B, 1)
    I_y = tf.expand_dims(I[:, 1], -1)
    I_z = tf.expand_dims(I[:, 2], -1)

    # 逐维编码
    x_enc = pre_encoder_keras(I_x, dims)  # (B, dims)
    y_enc = pre_encoder_keras(I_y, dims)
    z_enc = pre_encoder_keras(I_z, dims)

    if concat:
        # 按特征维拼接
        return tf.concat([x_enc, y_enc, z_enc], axis=2)   # (B, 3*dims)

    # 3D 外积：先各自扩维再逐元素乘
    x_enc = tf.reshape(x_enc, (-1, dims, 1, 1))  # (B, dims, 1, 1)
    y_enc = tf.reshape(y_enc, (-1, 1, dims, 1))  # (B, 1, dims, 1)
    z_enc = tf.reshape(z_enc, (-1, 1, 1, dims))  # (B, 1, 1, dims)

    I_encoded = x_enc * y_enc * z_enc            # (B, dims, dims, dims)
    return I_encoded
def vt_modle_hdc(x):
    m = 0.2 * tf.exp(-x ** 2 / 9.1)
    return (m / (1 + m) * x)


def vt_modle_gcn(x):
    m = 0.13 * tf.exp(-x ** 2 / 10.98)
    return (m / (1 + m) * x)

def find_short_delta(start, end, dims):
    delta = end - start
    delta = delta if abs(delta) < dims - abs(delta) else np.sign(delta) * (abs(delta) - dims)
    return delta

def find_short_vector(start, end, dims = 36, mode = "np"):
    delta = []
    for i,j in zip(start, end):
        delta.append(find_short_delta(i , j, dims))
    # 找到解码值指向实际值的最短向量
    if mode == "np":
        return np.array(delta)
    else:
        return tf.cast(delta , dtype = tf.float32)

def get_delta_hdc(start, end, dims = 36, mode = "np"):
    vec = find_short_vector(start, end, dims, mode)
    if mode == 'np':
        vector_norm = np.linalg.norm(vec)
        bais =np.array((0, 0)) if vector_norm > 9 else vt_modle_hdc(vector_norm) * vec / vector_norm
        bais = np.array(bais)
    else:
        vector_norm = tf.norm(vec)
        # 计算两种可能的情况
        zero_bais = tf.constant([0.0, 0.0], dtype=vec.dtype)
        scaled_bais = vt_modle_hdc(vector_norm) * vec / vector_norm

        # 根据条件选择结果
        bais = tf.cond(
            vector_norm > 9,
            lambda: zero_bais,
            lambda: scaled_bais
        )
    return bais

def get_delta_gcn(start, end, dims = 36):
    vec = find_short_vector(start, end, dims)
    vector_norm = np.linalg.norm(vec)
    bais = np.array((0, 0, 0)) if vector_norm > 9 else vt_modle_hdc(vector_norm) * vec / vector_norm

    return bais