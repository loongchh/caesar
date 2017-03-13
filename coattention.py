import logging

import tensorflow as tf

import util
import qa_data_util as du
from ops import *

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def assertion(var, var_name, expected):
    shape = var.get_shape().as_list()
    assert shape == expected, \
        "{} of incorrect shape. Expected {}, got {}".format(var_name, expected, shape)

class CoattentionModel():
    def __init__(self, embeddings, debug_shape=False):
        self.pretrained_embeddings = embeddings
        self.build(debug_shape)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.max_question_size),
                                                   name="question_placeholder")
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(FLAGS.batch_size, FLAGS.max_question_size),
                                                        name="question_mask_placeholder")
        self.question_seq_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size),
                                                       name="question_seq_placeholder")
        self.document_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                   name="document_placeholder")
        self.document_mask_placeholder = tf.placeholder(tf.bool, shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                        name="document_mask_placeholder")
        self.document_seq_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size),
                                                       name="document_seq_placeholder")
        self.span_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, 2),
                                               name="span_placeholder")
        self.answer_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.max_answer_size),
                                                 name="answer_placeholder")
        self.answer_mask_placeholder = tf.placeholder(tf.bool, shape=(FLAGS.batch_size, FLAGS.max_answer_size),
                                                      name="answer_mask_placeholder")
        self.answer_seq_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, ),
                                                     name="answer_seq_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")

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
        if data_batch['a'] is not None:
            feed_dict[self.answer_placeholder] = data_batch['a']
        if data_batch['a_m'] is not None:
            feed_dict[self.answer_mask_placeholder] = data_batch['a_m']
        if data_batch['a_s'] is not None:
            feed_dict[self.answer_seq_placeholder] = data_batch['a_s']

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.cast(tf.get_variable("embeddings", initializer=self.pretrained_embeddings, 
                                                 dtype=tf.float64), tf.float32)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)

        # assertion(question_embeddings, "question_embeddings", [FLAGS.batch_size, FLAGS.max_question_size + 1, None])
        # assertion(document_embeddings, "document_embeddings", [FLAGS.batch_size, FLAGS.max_document_size + 1, None])

        return question_embeddings, document_embeddings

    ## ==============================
    ## DOCUMENT AND QUESTION ENCODER
    def preprocessing(self, debug_shape=False):
        (Q_embed, D_embed) = self.add_embedding()

        # Encoding question and document.
        with tf.variable_scope("QD-ENCODE"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)
            (Q, _) = tf.nn.dynamic_rnn(cell, Q_embed, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            (D, _) = tf.nn.dynamic_rnn(cell, D_embed, dtype=tf.float32)
        
        assertion(Q, "Q", [FLAGS.batch_size, FLAGS.max_question_size, FLAGS.state_size])
        assertion(D, "D", [FLAGS.batch_size, FLAGS.max_document_size, FLAGS.state_size])

        # Add sentinel to the end of document/question.
        Q = tf.concat_v2([Q, tf.zeros([FLAGS.batch_size, 1, FLAGS.state_size])], 1)
        D = tf.concat_v2([D, tf.zeros([FLAGS.batch_size, 1, FLAGS.state_size])], 1)

        # Non-linear projection layer on top of the question encoding.
        with tf.variable_scope("Q-TANH"):
            W_q = tf.get_variable("W_q", shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_q = tf.get_variable("b_q", shape=(FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.constant_initializer(0.))
            Q = tf.tanh(tf.einsum('ijk,kl->ijl', Q, W_q) + b_q)

        assertion(Q, "Q", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.state_size])
        assertion(D, "D", [FLAGS.batch_size, FLAGS.max_document_size + 1, FLAGS.state_size])
        return (Q, D)

    ## ==============================
    ## COATTENTION ENCODER
    def encode(self, preprocessing, debug_shape=False):
        Q = preprocessing[0]
        D = preprocessing[1]

        # Affinity matrix.
        L = tf.batch_matmul(Q, tf.transpose(D, [0, 2, 1]))
        assertion(L, "L", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.max_document_size + 1])

        # Normalize with respect to question/document.
        Aq = tf.nn.softmax(L, dim=2)
        assertion(Aq, "Aq", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.max_document_size + 1])
        Ad = tf.transpose(tf.nn.softmax(L, dim=1), [0, 2, 1])
        assertion(Ad, "Ad", [FLAGS.batch_size, FLAGS.max_document_size + 1, FLAGS.max_question_size + 1])

        # Attention of the document w.r.t question.
        Cq = tf.matmul(Aq, D)
        assertion(Cq, "Cq", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.state_size])

        # Attention of previous attention w.r.t document, concatenated with attention of
        # question w.r.t. document.
        Cd = tf.concat_v2([tf.matmul(Ad, Q), tf.matmul(Ad, Cq)], 2)
        assertion(Cd, "Cd", [FLAGS.batch_size, FLAGS.max_document_size + 1, 2 * FLAGS.state_size])

        # Fusion of temporal information to the coattention context
        with tf.variable_scope("COATTENTION"):
            coatt = tf.concat_v2([D, Cd], 2)
            assertion(coatt, "coatt", [FLAGS.batch_size, FLAGS.max_document_size + 1, 3 * FLAGS.state_size])
            
            cell_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            (U, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, coatt, dtype=tf.float32, \
                sequence_length=self.document_seq_placeholder)
            U = tf.concat_v2(U, 2)
        
        assertion(U, "U", [FLAGS.batch_size, FLAGS.max_document_size + 1, 2 * FLAGS.state_size])
        return U


    ## ==============================
    ## DYNAMIC POINTING DECODER
    def decode(self, coattention, debug_shape=False):
        U = tf.transpose(coattention, [1, 0, 2])
        assertion(U, "U", [FLAGS.max_document_size + 1, FLAGS.batch_size, 2 * FLAGS.state_size])

        LSTM_dec = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.state_size)
        h = LSTM_dec.zero_state(FLAGS.state_size, dtype=tf.float32)
        HMN_a = highway_maxout(FLAGS.state_size, FLAGS.maxout_size)
        HMN_b = highway_maxout(FLAGS.state_size, FLAGS.maxout_size)

        # Get first estimated locations
        u_s = tf.gather_nd(U, list(zip(range(FLAGS.batch_size), [0] * FLAGS.batch_size)))
        u_e = tf.gather_nd(U, list(zip(range(FLAGS.batch_size), [0] * FLAGS.batch_size)))
        assertion(u_s, "u_s", [FLAGS.batch_size, 2 * FLAGS.state_size])
        assertion(u_e, "u_e", [FLAGS.batch_size, 2 * FLAGS.state_size])

        with tf.variable_scope('DECODER'):
            alpha = []
            beta = []

            for step in range(FLAGS.max_decode_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                
                # The dynamic decoder is a state machine
                (_, state) = tf.nn.rnn(LSTM_dec, [tf.concat_v2([u_s, u_e], 1)], dtype=tf.float32)
                h = tf.concat(1, state)
                # (_, h) = LSTM_dec(tf.concat_v2([u_s, u_e], 1), h)
                # assertion(h, "h", [FLAGS.state_size])
                batch_index = tf.to_int32(list(range(FLAGS.batch_size)))
                def select(pos, idx):
                    return tf.reshape(tf.gather(tf.gather(U, idx), tf.gather(pos, idx)), [-1])

                with tf.variable_scope('HIGHWAY-A'):
                    # Start score corresponding to each word in document
                    a = tf.map_fn(lambda u_t: HMN_a(u_t, h, u_s, u_e), U)

                    # Update current start position
                    s = tf.reshape(tf.argmax(a, 0), [FLAGS.batch_size])
                    assertion(s, "s", [FLAGS.batch_size])
                    # u_s = tf.gather_nd(U, list(zip(range(FLAGS.batch_size), s)))
                    u_s = tf.map_fn(lambda i: select(s, i), batch_index, dtype=tf.float32)
                    assertion(u_s, "u_s", [FLAGS.batch_size, 2 * FLAGS.state_size])

                with tf.variable_scope('HIGHWAY-B'):
                    # End score corresponding to each word in document
                    b = tf.map_fn(lambda u_t: HMN_b(u_t, h, u_s, u_e), U)

                    # Update current end position
                    e = tf.reshape(tf.argmax(b, 0), [FLAGS.batch_size])
                    assertion(e, "e", [FLAGS.batch_size])
                    # u_e = tf.gather_nd(U, list(zip(range(FLAGS.batch_size), e)))
                    u_e = tf.map_fn(lambda i: select(e, i), batch_index, dtype=tf.float32)
                    assertion(u_e, "u_e", [FLAGS.batch_size, 2 * FLAGS.state_size])

                alpha.append(tf.reshape(a, [FLAGS.batch_size, -1]))
                beta.append(tf.reshape(b, [FLAGS.batch_size, -1]))

        return (alpha, beta)

    def loss(self, decoded, debug_shape=False):
        alpha = decoded[0]
        beta = decoded[1]
        y = self.span_placeholder

        L1 = sum(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(a, y[:, 0]))
                 for a in alpha)
        L2 = sum(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(b, y[:, 1]))
                 for b in beta)
        return (L1 + L2)

    def add_training_op(self, loss, debug_shape=False):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss[0])
        return (train_op,) + loss

    def build(self, debug_shape):
        self.add_placeholders()
        self.preprocessed = self.preprocessing(debug_shape)
        self.encoded = self.encode(self.preprocessed, debug_shape)
        self.decoded = self.decode(self.encoded, debug_shape)
        self.loss = self.loss(self.decoded, debug_shape)
        self.train_op = self.add_training_op(self.loss, debug_shape)

    def debug_shape(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op_output = sess.run(
            fetches = util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )
        logger.info("loss: {}".format(train_op_output[1]))
        # logger.info("betas: {}".format(train_op_output[2]))
        logger.info("pred: {}".format(train_op_output[3]))

        for i, tensor in enumerate(self.train_op):
            if tensor.name.startswith("debug_"):
                logger.debug("Shape of {} == {}".format(tensor.name[6:], train_op_output[i]))

    def predict_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        decoded = sess.run(
            fetches = util.tuple_to_list(*self.decoded),
            feed_dict=feed
        )
        pred = du.get_answer_from_span(decoded[1])
        return pred

    def train_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op = sess.run(
            fetches = util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )

        loss = train_op[1]
        pred = du.get_answer_from_span(train_op[3])

        return loss, pred
