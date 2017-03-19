import logging

import tensorflow as tf

import util
from qa_data_util import get_answer_from_span
# from ops import *

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def assert_shape(var, var_name, expected):
    shape = var.get_shape().as_list()
    assert shape == expected, \
        "{} of incorrect shape. Expected {}, got {}".format(var_name, expected, shape)

class CoattentionModel():
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
        self.sentence_span_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.max_document_size, 2),
                                                        name="sentence_span_placeholder")
        self.sentence_number_placeholder = tf.placeholder(tf.int32, shape=(None),
                                                          name="sentence_number_placeholder")
        self.answer_sentence_placeholder = tf.placeholder(tf.int32, shape=(None),
                                                          name="answer_sentence_placeholder")
        self.span_placeholder = tf.placeholder(tf.int32, shape=(None, 2),
                                               name="span_placeholder")
        self.answer_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.max_answer_size),
                                                 name="answer_placeholder")
        self.answer_mask_placeholder = tf.placeholder(tf.bool, shape=(None, FLAGS.max_answer_size),
                                                      name="answer_mask_placeholder")
        self.answer_seq_placeholder = tf.placeholder(tf.int32, shape=[None],
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
        if 's_s' in data_batch and data_batch['s_s'] is not None:
            feed_dict[self.sentence_span_placeholder] = data_batch['s_s']
        if 's_n' in data_batch and data_batch['s_n'] is not None:
            feed_dict[self.sentence_number_placeholder] = data_batch['s_n']
        if 'an_s' in data_batch and data_batch['an_s'] is not None:
            feed_dict[self.answer_sentence_placeholder] = data_batch['an_s']
        if 's' in data_batch and data_batch['s'] is not None:
            feed_dict[self.span_placeholder] = data_batch['s']
        if 'a' in data_batch and data_batch['a'] is not None:
            feed_dict[self.answer_placeholder] = data_batch['a']
        if 'a_m' in data_batch and data_batch['a_m'] is not None:
            feed_dict[self.answer_mask_placeholder] = data_batch['a_m']
        if 'a_s' in data_batch and data_batch['a_s'] is not None:
            feed_dict[self.answer_seq_placeholder] = data_batch['a_s']

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=FLAGS.embedding_trainable)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)
        return question_embeddings, document_embeddings

    ## ==============================
    ## DOCUMENT AND QUESTION ENCODER
    def contextual_preprocessing(self, debug=False):
        
        def summarize((document, question, sentences, n_sen, d_len, q_len)):
            assert_shape(document, "document", [FLAGS.max_document_size, FLAGS.state_size])
            assert_shape(question, "question", [FLAGS.max_question_size, FLAGS.state_size])
            assert_shape(sentences, "sentences", [FLAGS.max_document_size, 2])
            # assert_shape(n_sen, "n_sen", [None])
            # assert_shape(q_len, "q_len", [None])

            # Question sentence representation
            if FLAGS.pool_type.lower() == "max":
                q_rep = tf.reduce_max(question[:q_len, :], axis=0)
            else:
                q_rep = tf.reduce_mean(question[:q_len, :], axis=0)
            assert_shape(q_rep, "q_rep", [FLAGS.state_size])
            
            def process_sentence(sen_idx):
                sen = document[sen_idx[0]:sen_idx[1], :]
                rep = tf.reduce_max(sen, axis=0) if FLAGS.pool_type.lower() == "max" \
                                                 else tf.reduce_mean(sen, axis=0)
                rep /= tf.sqrt(tf.reduce_sum(tf.square(sen)))  # normalized sentence rep
                assert_shape(rep, "rep", [FLAGS.state_size])  
                sim = tf.reduce_sum(rep * q_rep)  # cosine similarity
                return sim

            # Get sentence-level representation and sentence length
            sen_sim = tf.map_fn(process_sentence, sentences[:n_sen, ], dtype=tf.float32)
            assert_shape(sen_sim, "sen_sim", [None])

            # Find key sentence with highest similarity to question
            (_, key_sen) = tf.nn.top_k(sen_sim)
            key_sen = key_sen[0]

            # Truncate document around the key sentence
            key_sen_from = sentences[key_sen, 0]
            key_sen_to = sentences[key_sen, 1]
            key_sen_cen = (key_sen_from + key_sen_to) / 2
            
            # Make sure the summary falls within actual document
            doc_to = key_sen_cen + FLAGS.max_summary_size / 2
            doc_to = tf.cond(tf.greater_equal(doc_to, d_len), lambda: d_len, lambda: doc_to)
            doc_from = doc_to - FLAGS.max_summary_size
            doc_from = tf.cond(tf.less(doc_from, 0), lambda: tf.constant(0), lambda: doc_from)
            
            return (document[doc_from:(doc_from + FLAGS.max_summary_size), :], doc_from)

        (Q_embed, D_embed) = self.add_embedding()
        summary_start = tf.constant(0)

        # Encoding question and document.
        with tf.variable_scope("QD-ENCODE"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)
            (Q, _) = tf.nn.dynamic_rnn(cell, Q_embed, dtype=tf.float32,
                                       sequence_length=self.question_seq_placeholder)
            tf.get_variable_scope().reuse_variables()
            (D, _) = tf.nn.dynamic_rnn(cell, D_embed, dtype=tf.float32,
                                       sequence_length=self.document_seq_placeholder)
        
        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])
        assert_shape(D, "D", [None, FLAGS.max_document_size, FLAGS.state_size])

        if FLAGS.max_summary_size < FLAGS.max_document_size:
            # Summarized document matrix of size max_summary_size
            (D, summary_start) = tf.map_fn(summarize, (D, Q, self.sentence_span_placeholder, \
                self.sentence_number_placeholder, self.document_seq_placeholder, self.question_seq_placeholder), \
                dtype=(tf.float32, tf.int32), back_prop=False)
            # assert_shape(D, "D", [None, FLAGS.max_summary_size, FLAGS.state_size])

        # Non-linear projection layer on top of the question encoding.
        with tf.variable_scope("Q-TANH"):
            W_q = tf.get_variable("W_q", shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_q = tf.get_variable("b_q", shape=(FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.constant_initializer(0.))
            Q = tf.tanh(tf.map_fn(lambda x: tf.matmul(x, W_q) + b_q, Q))

        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])
        return (Q, D, summary_start)

    ## ==============================
    ## COATTENTION ENCODER
    def coattention_encode(self, preprocessing, debug=False):
        Q = preprocessing[0]
        D = preprocessing[1]
        summary_start = preprocessing[2]

        # Affinity matrix.
        L = tf.batch_matmul(Q, tf.transpose(D, [0, 2, 1]))
        # assert_shape(L, "L", [None, FLAGS.max_question_size, FLAGS.max_summary_size])

        # Normalize with respect to question/document.
        A_q = tf.map_fn(lambda x: tf.nn.softmax(x, dim=0), L, dtype=tf.float32)
        # assert_shape(A_q, "A_q", [None, FLAGS.max_question_size, FLAGS.max_summary_size])
        A_d = tf.map_fn(lambda x: tf.nn.softmax(x, dim=0), tf.transpose(L, [0, 2, 1]), dtype=tf.float32)
        # assert_shape(A_d, "A_d", [None, FLAGS.max_summary_size, FLAGS.max_question_size])

        # Attention of the document w.r.t question.
        C_q = tf.batch_matmul(A_q, D)
        assert_shape(C_q, "C_q", [None, FLAGS.max_question_size, FLAGS.state_size])

        # Attention of previous attention w.r.t document, concatenated with attention of
        # question w.r.t. document.
        C_d = tf.concat(2, [tf.batch_matmul(A_d, Q), tf.batch_matmul(A_d, C_q)])
        # assert_shape(C_d, "C_d", [None, FLAGS.max_summary_size, 2 * FLAGS.state_size])

        # Fusion of temporal information to the coattention context
        with tf.variable_scope("COATTENTION"):
            coatt = tf.concat(2, [D, C_d])
            # assert_shape(coatt, "coatt", [None, FLAGS.max_summary_size, 3 * FLAGS.state_size])
            
            cell_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            (U, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, coatt, dtype=tf.float32, \
                sequence_length=self.document_seq_placeholder - summary_start)
            U = tf.concat(2, U)
        
        # assert_shape(U, "U", [None, FLAGS.max_summary_size, 2 * FLAGS.state_size])
        return (U, summary_start)

    ## ==============================
    ## FEED FORWARD DECODER
    def feed_forward_decode(self, encode, debug=False):
        Hr = encode[0]
        summary_start = encode[1]

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
            Hr_W1 = tf.reshape(Hr_W1, [-1, FLAGS.max_summary_size, 2])
            h = tf.transpose(Hr_W1 + b1, perm = [0,2,1])
            betas = tf.nn.softmax(h)
            pred = tf.argmax(betas, 2)

        return (h, pred, summary_start)

    ## ==============================
    ## ANSWER POINTER DECODER
    def answer_pointer_decode(self, encode, debug=False):
        H_r = encode[0]
        summary_start = encode[1]
        # assert_shape(H_r, "H_r", [None, FLAGS.max_summary_size, 2 * FLAGS.state_size])

        with tf.variable_scope("answer_pointer_decode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.state_size, state_is_tuple=True)
            ha = cell.zero_state(tf.shape(H_r)[0], tf.float32)
            assert_shape(ha[1], "ha[1]", [None, FLAGS.state_size])
            beta = []

            V = tf.get_variable('V', shape=(2 * FLAGS.state_size, FLAGS.state_size),
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            W_a = tf.get_variable('W_a', shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_a = tf.get_variable('b_a', shape=(FLAGS.state_size), dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.))
            v = tf.get_variable('v', shape=(1, FLAGS.state_size), dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            c = tf.get_variable('c', shape=(1, 1), dtype=tf.float32, initializer=tf.constant_initializer(0.))

            for k in range(2):
                if k > 0:
                    tf.get_variable_scope().reuse_variables()

                VH_r = tf.map_fn(lambda x: tf.matmul(x, V), H_r)
                # VH_r = tf.einsum('ijk,kl->ijl', H_r, V)
                # assert_shape(VH_r, "VH_r", [None, FLAGS.max_summary_size, FLAGS.state_size])
                W_aH_ab_a = tf.matmul(ha[1], W_a) + b_a
                assert_shape(W_aH_ab_a, "W_aH_ab_a", [None, FLAGS.state_size])
                W_aH_ab_a = tf.expand_dims(W_aH_ab_a, axis=1)
                F_k = tf.nn.tanh(VH_r + tf.tile(W_aH_ab_a, [1, FLAGS.max_summary_size, 1]))
                F_k = tf.transpose(F_k, perm=[0, 2, 1])
                # assert_shape(F_k, "F_k", [None, FLAGS.state_size, FLAGS.max_summary_size])
                
                v_tF_k = tf.map_fn(lambda x: tf.matmul(v, x), F_k)
                # v_tF_k = tf.einsum('ij,kjl->kil', v, F_k)
                # assert_shape(v_tF_k, "v_tF_k", [None, 1, FLAGS.max_summary_size])
                beta_no_softmax = v_tF_k + tf.tile(c, [1, FLAGS.max_summary_size])
                beta_k = tf.nn.softmax(beta_no_softmax)
                # assert_shape(beta_k, "beta_k", [None, 1, FLAGS.max_summary_size])
                # assert_shape(v_tF_k, "v_tF_k", [None, 1, FLAGS.max_summary_size])
                beta_k = v_tF_k + tf.tile(c, [1, FLAGS.max_summary_size])
                # assert_shape(beta_k, "beta_k", [None, 1, FLAGS.max_summary_size])

                H_rbeta_k = tf.squeeze(tf.batch_matmul(beta_k, H_r), squeeze_dims=1)
                assert_shape(H_rbeta_k, "H_rbeta_k", [None, 2 * FLAGS.state_size])

                beta.append(beta_no_softmax)
                (_, ha) = cell(H_rbeta_k, ha)

            beta = tf.concat(1, beta)
            # assert_shape(beta, "beta", [None, 2, FLAGS.max_summary_size])
            pred = tf.to_int32(tf.argmax(beta, axis=2))
            if FLAGS.max_summary_size < FLAGS.max_document_size:
                pred += tf.tile(tf.expand_dims(summary_start, -1), [1, 2])

        return (beta, pred, summary_start)

    def cross_entropy_loss(self, decode, debug=False):
        beta = decode[0]
        summary_start = decode[2]
        s = self.span_placeholder[:, 0]
        e = self.span_placeholder[:, 1]
        retain = tf.constant(0)
        
        s -= summary_start  # no change without summarization
        e -= summary_start

        # Mask determine if answer span falls within the summary
        mask = tf.logical_and(tf.greater_equal(s, 0), tf.less(e, FLAGS.max_summary_size))
        retain = tf.count_nonzero(mask)  # answer is retained in summary
        beta = tf.boolean_mask(beta, mask)
        s = tf.boolean_mask(s, mask)
        e = tf.boolean_mask(e, mask)

        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 0, :], s))
        L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 1, :], e))
        return ((L1 + L2) / 2., retain)

    def add_train_op(self, loss, debug=False):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gradients = optimizer.compute_gradients(loss[0])
        (grad, var) = zip(*gradients)
        (grad, _) = tf.clip_by_global_norm(grad, FLAGS.max_gradient_norm)
        
        grad_norm = []
        for (i, v) in enumerate(var):
            grad_norm.append(tf.global_norm([grad[i]]))
        grad_norm = tf.pack(grad_norm)

        train_op = optimizer.apply_gradients(zip(grad, var))
        return (train_op, grad_norm, loss[0], loss[1])

    def _build(self, debug):
        self.add_placeholders()
        self.preprocessing = self.contextual_preprocessing(debug)
        self.encode = self.coattention_encode(self.preprocessing, debug)
        self.decode = self.answer_pointer_decode(self.encode, debug)
        # self.decode = self.feed_forward_decode(self.encode, debug)
        self.loss = self.cross_entropy_loss(self.decode, debug)
        self.train_op = self.add_train_op(self.loss, debug)

    def debug(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        debug_output = sess.run(util.tuple_to_list(*self.train_op), feed_dict=feed)

        logger.debug("Gradient {}".format(debug_output[1]))
        logger.debug("Loss {}".format(debug_output[2]))
        # logger.debug("pred: {}".format(debug_output[4]))
        # logger.debug(debug_output)
        # for i, tensor in enumerate(self.decode):
        #     if tensor.name.startswith("debug_"):
        #         logger.debug("Shape of {} == {}".format(tensor.name[6:], debug_output[i]))

    def summary_success(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        debug_output = sess.run(util.tuple_to_list(*self.train_op), feed_dict=feed)
        return debug_output[3]

    def predict_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        decode_output = sess.run(util.tuple_to_list(*self.decode), feed_dict=feed)

        pred = get_answer_from_span(decode_output[1])
        return pred

    def train_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        train_op_output = sess.run(util.tuple_to_list(*self.train_op), feed_dict=feed)
        
        grad_norm = train_op_output[1]
        loss = train_op_output[2]
        retain = train_op_output[3]
        return grad_norm, loss, retain
