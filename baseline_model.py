import logging
import tensorflow as tf
from model import QAModel
import util

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class BaselineModel(QAModel):

    def __init__(self, embeddings, debug_shape=False):
        super(BaselineModel, self).__init__(embeddings, debug_shape)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, None),
                                                name="question_placeholder")
        self.question_mask_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, None),
                                                name="question_placeholder")
        self.document_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, None),
                                                name="document_placeholder")
        self.span_placeholder = tf.placeholder(tf.int32,
                                                 shape=(FLAGS.batch_size, 2),
                                                 name="span_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  name="dropout_placeholder")


    def create_feed_dict(self, question_batch, document_batch, span_batch=None, dropout=1):
        feed_dict = {
            self.question_placeholder: question_batch,
            self.document_placeholder: document_batch,
        }

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        if span_batch is not None:
            feed_dict[self.span_placeholder] = span_batch

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.cast(
            tf.get_variable("all_embeddings", initializer=self.pretrained_embeddings, dtype=tf.float64),
            tf.float32)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)

        return question_embeddings, document_embeddings

    def add_encoder_op(self, debug_shape=False):
        q_input,d_input = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        if debug_shape:
            return (
                tf.shape(d_input,name="debug_d"),
                tf.shape(q_input,name="debug_q"),
            )
        return d_input,

