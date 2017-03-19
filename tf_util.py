import tensorflow as tf
import numpy as np

def _3d_X_2d(_3d_tensor, _2d_tensor):
    shape_3d = _3d_tensor.get_shape().as_list()
    shape_2d = _2d_tensor.get_shape().as_list()

    return tf.reshape(
        tf.matmul(
            tf.reshape(
                _3d_tensor,
                [-1, shape_2d[0]]
            ),
        _2d_tensor),
        [-1, shape_3d[1], shape_2d[1]]
    )

def test_3d_X_2d():
    L = tf.constant(np.random.rand(10,100,300))
    W = tf.constant(np.random.rand(300,300))
    with tf.Session() as session:
        a = session.run(tf.map_fn(lambda x: tf.matmul(x, W), L))
        b = session.run(_3d_X_2d(L,W))
        assert np.all(a==b)


def assert_shape(var, var_name, expected):
    shape = var.get_shape().as_list()
    assert shape == expected, \
        "{} of incorrect shape. Expected {}, got {}".format(var_name, expected, shape)

if __name__ == '__main__':
    test_3d_X_2d()
