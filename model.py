"""
A model for Question Answer system
"""
import logging

logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class QAModel(object):
    def __init__(self, pretrained_embeddings, debug_shape=False):
        self.pretrained_embeddings = pretrained_embeddings
        self.build(debug_shape)

    def assert_shape(self, var, var_name, expected):
        shape = var.get_shape().as_list()
        assert shape == expected, "{} of incorrect shape. Expected {}, got {}".format(var_name, expected, shape)

    def debug_shape(self, sess, data_batch):
        raise NotImplementedError

    def predict_on_batch(self, sess, data_batch):
        raise NotImplementedError

    def train_on_batch(self, sess, data_batch):
        raise NotImplementedError