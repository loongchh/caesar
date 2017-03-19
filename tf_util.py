import tensorflow as tf
import numpy as np

def _3d_X_2d(_3d_tensor, _2d_tensor):
    return tf.reshape(
        tf.matmul(
            tf.reshape(
                _3d_tensor,
                [-1, tf.shape(_2d_tensor)[0]]
            ),
        _2d_tensor ),
        [-1, tf.shape(_3d_tensor)[1], tf.shape(_3d_tensor)[2]]
    )

def test_3d_X_2d():
    L = np.random.rand(10,100,300)
    W = np.random.rand(300,300)
    with tf.Session() as session:
        a = session.run(tf.map_fn(lambda x: tf.matmul(x, W), L))
        b = session.run(_3d_X_2d(L,W))
        print np.all(a==b)

def assert_shape(var, var_name, expected):
    shape = var.get_shape().as_list()
    assert shape == expected, \
        "{} of incorrect shape. Expected {}, got {}".format(var_name, expected, shape)

if __name__ == '__main__':
    test_3d_X_2d()
