import logging

import tensorflow as tf

from code.model import QAModel

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class AttentionModel(QAModel):

    def __init__(self, embeddings, debug_shape=False):
        super(BaselineModel, self).__init__(embeddings, debug_shape)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, FLAGS.max_question_size),
                                                name="question_placeholder")
        self.question_mask_placeholder = tf.placeholder(tf.bool,
                                                shape=(FLAGS.batch_size, FLAGS.max_question_size),
                                                name="question_mask_placeholder")

        self.question_seq_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size),
                                                name="question_seq_placeholder")

        self.document_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                name="document_placeholder")

        self.document_mask_placeholder = tf.placeholder(tf.bool,
                                                shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                name="document_mask_placeholder")

        self.document_seq_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size),
                                                name="document_seq_placeholder")

        self.span_placeholder = tf.placeholder(tf.int32,
                                                 shape=(FLAGS.batch_size, 2),
                                                 name="span_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  name="dropout_placeholder")

    def create_feed_dict(self, data_batch, dropout=1):
        feed_dict = {
            self.question_placeholder: data_batch['q'],
            self.question_mask_placeholder: data_batch['q_m'],
            self.question_seq_placeholder: data_batch['q_s'],
            self.document_placeholder: data_batch['c'],
            self.document_mask_placeholder: data_batch['c_m'],
            self.document_seq_placeholder: data_batch['c_s']
        }

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        if data_batch['s'] is not None:
            feed_dict[self.span_placeholder] = data_batch['s']

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.cast(
            tf.get_variable("embeddings", initializer=self.pretrained_embeddings, dtype=tf.float64),
            tf.float32)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)

        return question_embeddings, document_embeddings

    def add_encoder_op(self, debug_shape=False):
        q_input,d_input = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("BLSTM"):

            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)

            initial_state_fw = cell_fw.zero_state(FLAGS.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(FLAGS.batch_size, tf.float32)

            ( (output_fw,output_bw), ((_,states_fw), (_,states_bw))) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                           cell_bw=cell_bw,
                                                           inputs=q_input,
                                                           initial_state_fw=initial_state_fw,
                                                           initial_state_bw=initial_state_bw,
                                                           sequence_length=self.question_seq_placeholder
                                                           )
            q_representation = tf.concat(concat_dim=1, values=[states_fw,states_bw])
            # q_representation = tf.nn.dropout(q_representation, dropout_rate)

        if debug_shape:
            return (
                tf.shape(d_input,name="debug_d_input"),
                tf.shape(q_input,name="debug_q_input"),
                tf.shape(output_fw,name="debug_output_fw"),
                tf.shape(output_bw,name="debug_output_bw"),
                tf.shape(states_fw,name="debug_states_fw"),
                tf.shape(states_bw,name="debug_states_bw"),
                tf.shape(q_representation,name="debug_q_representation")
            )
        return d_input,

