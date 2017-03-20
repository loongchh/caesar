import logging

import tensorflow as tf

import util
from qa_data_util import get_answer_from_span
from tf_util import _3d_X_2d
from tf_util import assert_shape
# from ops import *

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class CoattentionWithoutSummaryModel():
    def __init__(self, embeddings, debug=False):
        self.pretrained_embeddings = embeddings
        self._build(debug)

    def add_placeholders(self):
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.max_question_size),
                                                   name="question_placeholder")
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, FLAGS.max_question_size),
                                                        name="question_mask_placeholder")
        self.question_seq_placeholder = tf.placeholder(tf.int32, shape=[None],
                                                       name="question_seq_placeholder")
        self.document_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.max_document_size),
                                                   name="document_placeholder")
        self.document_mask_placeholder = tf.placeholder(tf.bool, shape=(None, FLAGS.max_document_size),
                                                        name="document_mask_placeholder")
        self.document_seq_placeholder = tf.placeholder(tf.int32, shape=[None], name="document_seq_placeholder")

        self.span_placeholder = tf.placeholder(tf.int32, shape=(None, 2),
                                               name="span_placeholder")

        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")

    def create_feed_dict(self, data_batch, dropout=1.):
        feed_dict = {
            self.dropout_placeholder: dropout,
            self.question_placeholder: data_batch['q'],
            self.question_mask_placeholder: data_batch['q_m'],
            self.question_seq_placeholder: data_batch['q_s'],
            self.document_placeholder: data_batch['c'],
            self.document_mask_placeholder: data_batch['c_m'],
            self.document_seq_placeholder: data_batch['c_s']
        }

        if 's' in data_batch and data_batch['s'] is not None:
            feed_dict[self.span_placeholder] = data_batch['s']

        return feed_dict

    def add_embedding(self):
        # all_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=FLAGS.embedding_trainable)
        all_embeddings = tf.constant(self.pretrained_embeddings)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)
        return question_embeddings, document_embeddings

    ## ==============================
    ## DOCUMENT AND QUESTION ENCODER
    def contextual_preprocessing(self, debug=False):
        (Q_embed, D_embed) = self.add_embedding()

        # Encoding question and document.
        with tf.variable_scope("QD-ENCODE"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
            (Q, _) = tf.nn.dynamic_rnn(cell_fw, Q_embed, sequence_length=self.question_seq_placeholder, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            (D, _) = tf.nn.dynamic_rnn(cell_fw, D_embed, sequence_length=self.document_seq_placeholder, dtype=tf.float32)
        
        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])
        assert_shape(D, "D", [None, FLAGS.max_document_size, FLAGS.state_size])

        # Non-linear projection layer on top of the question encoding.
        with tf.variable_scope("Q-TANH"):
            W_q = tf.get_variable("W_q", shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_q = tf.get_variable("b_q", shape=(FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.constant_initializer(0.))
            Q = tf.tanh(_3d_X_2d(Q, W_q) + b_q)

        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])

        return Q, D


    ## ==============================
    ## COATTENTION ENCODER
    def coattention_encode(self, preprocessing, debug=False):
        Q = preprocessing[0]
        D = preprocessing[1]

        # Affinity matrix.
        L = tf.batch_matmul(Q, tf.transpose(D, [0, 2, 1]))
        assert_shape(L, "L", [None, FLAGS.max_question_size, FLAGS.max_document_size])

        # Normalize with respect to question/document.
        A_q = tf.nn.softmax(L)
        assert_shape(A_q, "A_q", [None, FLAGS.max_question_size, FLAGS.max_document_size])
        A_d = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]))
        assert_shape(A_d, "A_d", [None, FLAGS.max_document_size, FLAGS.max_question_size])

        # Attention of the document w.r.t question.
        C_q = tf.batch_matmul(A_q, D)
        assert_shape(C_q, "C_q", [None, FLAGS.max_question_size, FLAGS.state_size])

        # Attention of previous attention w.r.t document, concatenated with attention of
        # question w.r.t. document.
        C_d = tf.concat(2, [tf.batch_matmul(A_d, Q), tf.batch_matmul(A_d, C_q)])
        assert_shape(C_d, "C_d", [None, FLAGS.max_document_size, 2 * FLAGS.state_size])

        # Fusion of temporal information to the coattention context
        with tf.variable_scope("COATTENTION"):
            coatt = tf.concat(2, [D, C_d])
            assert_shape(coatt, "coatt", [None, FLAGS.max_document_size, 3 * FLAGS.state_size])
            
            cell_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)

            cell_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
            (U, _) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                coatt,
                dtype=tf.float32,
                sequence_length=self.document_seq_placeholder
            )
            U = tf.concat(2, U)
        
        return U, A_q, A_d

    ## ==============================
    ## FEED FORWARD DECODER
    def feed_forward_decode(self, encode, debug=False):
        Hr = encode[0]

        with tf.variable_scope("Feed_Forward_Prediction"):
            W1 =tf.get_variable(name='W1',
                               shape = [2*FLAGS.state_size, 2],
                               dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.1)
                                # initializer=tf.contrib.layers.xavier_initializer()
                               )

            b1 =tf.get_variable(name='b1',
                                 shape = [2],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            Hr_W1 = tf.matmul(tf.reshape(Hr, [-1, 2 * FLAGS.state_size]), W1)
            Hr_W1 = tf.reshape(Hr_W1, [-1, FLAGS.max_document_size, 2])
            h = tf.transpose(Hr_W1 + b1, perm = [0,2,1])
            betas = tf.nn.softmax(h)
            pred = tf.argmax(betas, 2)

        return (h, pred, ) + encode[1:]

    ## ==============================
    ## ANSWER POINTER DECODER
    def answer_pointer_decode(self, encode, debug=False):
        H_r = encode[0]

        assert_shape(H_r, "H_r", [None, FLAGS.max_document_size, 2 * FLAGS.state_size])

        with tf.variable_scope("answer_pointer_decode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.state_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)

            ha = cell.zero_state(tf.shape(H_r)[0], tf.float32)
            assert_shape(ha[1], "ha[1]", [None, FLAGS.state_size])
            beta = []

            V = tf.get_variable('V', shape=(2 * FLAGS.state_size, FLAGS.state_size),
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_a = tf.get_variable('W_a', shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_a = tf.get_variable('b_a', shape=(FLAGS.state_size), dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.))
            v = tf.get_variable('v', shape=(FLAGS.state_size, 1), dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            c = tf.get_variable('c', shape=(1, 1), dtype=tf.float32, initializer=tf.constant_initializer(0.))

            for k in range(2):
                if k > 0:
                    tf.get_variable_scope().reuse_variables()

                VH_r = _3d_X_2d(H_r, V)
                assert_shape(VH_r, "VH_r", [None, FLAGS.max_document_size, FLAGS.state_size])
                W_aH_ab_a = tf.matmul(ha[1], W_a) + b_a
                assert_shape(W_aH_ab_a, "W_aH_ab_a", [None, FLAGS.state_size])
                W_aH_ab_a = tf.expand_dims(W_aH_ab_a, axis=1)
                F_k = tf.nn.tanh(VH_r + tf.tile(W_aH_ab_a, [1, FLAGS.max_document_size, 1]))
                # F_k = tf.transpose(F_k, perm=[0, 2, 1])
                assert_shape(F_k, "F_k", [None, FLAGS.max_document_size, FLAGS.state_size])
                
                v_tF_k = _3d_X_2d(F_k, v)
                v_tF_k = tf.transpose(_3d_X_2d(F_k, v), perm=[0,2,1])
                assert_shape(v_tF_k, "v_tF_k", [None, 1, FLAGS.max_document_size])

                beta_no_softmax = v_tF_k + tf.tile(c, [1, FLAGS.max_document_size])
                beta_k = tf.nn.softmax(beta_no_softmax)
                assert_shape(beta_k, "beta_k", [None, 1, FLAGS.max_document_size])

                H_rbeta_k = tf.squeeze(tf.batch_matmul(beta_k, H_r), squeeze_dims=1)
                assert_shape(H_rbeta_k, "H_rbeta_k", [None, 2 * FLAGS.state_size])

                beta.append(beta_no_softmax)
                (_, ha) = cell(H_rbeta_k, ha)

            beta = tf.concat(1, beta)
            assert_shape(beta, "beta", [None, 2, FLAGS.max_document_size])
            pred = tf.to_int32(tf.argmax(beta, axis=2))

        return (beta, pred, ) + encode[1:]

    def cross_entropy_loss(self, decode, debug=False):
        beta = decode[0]

        s = self.span_placeholder[:, 0]
        e = self.span_placeholder[:, 1]

        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 0, :], s))
        L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 1, :], e))
        return (L1 + L2) / 2.,

    def add_train_op(self, loss, debug=False):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gradients = optimizer.compute_gradients(loss[0])
        (grad, var) = zip(*gradients)
        (grad, _) = tf.clip_by_global_norm(grad, FLAGS.max_gradient_norm)
        
        grad_norm = []
        for (i, v) in enumerate(var):
            logger.info(v)
            grad_norm.append(tf.global_norm([grad[i]]))
        grad_norm = tf.pack(grad_norm)

        train_op = optimizer.apply_gradients(zip(grad, var))
        return train_op, grad_norm, loss[0]

    def _build(self, debug):
        self.add_placeholders()
        self.preprocessing = self.contextual_preprocessing(debug)
        self.encode = self.coattention_encode(self.preprocessing, debug)
        self.decode = self.answer_pointer_decode(self.encode, debug)
        # self.decode = self.feed_forward_decode(self.encode, debug)
        self.loss = self.cross_entropy_loss(self.decode, debug)
        self.train_op = self.add_train_op(self.loss, debug)

    def debug(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch, dropout=FLAGS.dropout)
        debug_output = sess.run(util.tuple_to_list(*self.train_op), feed_dict=feed)

        logger.debug("Gradient {}".format(debug_output[1]))
        logger.debug("Loss {}".format(debug_output[2]))
        # logger.debug("pred: {}".format(debug_output[4]))
        # logger.debug(debug_output)
        # for i, tensor in enumerate(self.decode):
        #     if tensor.name.startswith("debug_"):
        #         logger.debug("Shape of {} == {}".format(tensor.name[6:], debug_output[i]))

    def predict_on_batch(self, sess, data_batch, rev_vocab=None):
        feed = self.create_feed_dict(data_batch, dropout=1.0)
        decode_output = sess.run(util.tuple_to_list(*self.decode), feed_dict=feed)

        pred = get_answer_from_span(decode_output[1])
        A_q = decode_output[2]
        A_d = decode_output[3]
        # self.plot_attention_matrix(A_q, A_d, data_batch, rev_vocab, pred)
        return pred

    def plot_attention_matrix(self, A_q, A_d, data_batch, rev_vocab, pred):


        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('attention_plot.pdf') as pdf:
            plt.matshow(A_q[0], fignum=10, cmap=plt.cm.gray)
            pdf.savefig()
            plt.matshow(np.transpose(A_d[0]), fignum=10, cmap=plt.cm.gray)
            pdf.savefig()

            fig = plt.figure()
            fig.text(.1,.3," ".join([ rev_vocab[int(id)] if i%10!=0 else "\n"+rev_vocab[int(id)] for i,id in enumerate(data_batch['c'][0]) if id!=0]))
            pdf.savefig()
            fig = plt.figure()
            fig.text(.1,.9," ".join([ rev_vocab[int(id)] if i%10!=0 else "\n"+ rev_vocab[int(id)] for i,id in enumerate(data_batch['q'][0])if id!=0]))
            pdf.savefig()
            plt.close()
        logger.info(" ".join([rev_vocab[int(data_batch['c'][0][int(id)])] for id in pred[0]]))
        logger.info(A_q[0])

    def train_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch, dropout=FLAGS.dropout)
        train_op_output = sess.run(util.tuple_to_list(*self.train_op), feed_dict=feed)
        
        grad_norm = train_op_output[1]
        loss = train_op_output[2]
        return grad_norm, loss, 0
