"""
A model for Question Answer system
"""
import logging

import tensorflow as tf
import code.util

FLAGS = tf.app.flags.FLAGS

logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class QAModel(object):
    def __init__(self, pretrained_embeddings, debug_shape=False):
        self.pretrained_embeddings = pretrained_embeddings
        self.build(debug_shape)

    def build(self, debug_shape):
        self.add_placeholders()
        self.encoded_representation = self.add_encoder_op(debug_shape)
        self.decoded_representation = self.add_decoder_op(self.encoded_representation, debug_shape)
        self.loss = self.add_loss_op(self.decoded_representation, debug_shape)
        self.train_op = self.add_training_op(self.loss, debug_shape)

    def add_placeholders(self):
        raise NotImplementedError

    def create_feed_dict(self, data_batch, dropout=1):
        raise NotImplementedError

    def add_encoder_op(self, debug_shape=False):
        raise NotImplementedError

    def add_decoder_op(self, encoded_representation, debug_shape=False):
        return encoded_representation

    def add_loss_op(self, decoded_representation, debug_shape=False):
        return decoded_representation

    def add_training_op(self, loss, debug_shape=False):
        return loss


## Execution Methods ------------------------------------------

    def debug_shape(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op_output = sess.run(
            fetches = code.util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )
        for i, tensor in enumerate(self.train_op):
            if tensor.name.startswith("debug_"):
                logger.debug("Shape of {} == {}".format(tensor.name[6:], train_op_output[i]))

    def predict_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        decoded_representation = sess.run(
            fetches = code.util.tuple_to_list(*self.decoded_representation),
            feed_dict=feed
        )
        pred = decoded_representation[0]
        return pred

    def train_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op = sess.run(
            fetches = code.util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )

        loss = train_op[1]
        pred = train_op[2]

        return loss, pred