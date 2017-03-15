import logging

import tensorflow as tf

import util
import qa_data_util as du
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
    def __init__(self, embeddings, debug_shape=False):
        self.pretrained_embeddings = embeddings
        self.build(debug_shape)

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
        self.document_seq_placeholder = tf.placeholder(tf.int32, shape=(None), name="document_seq_placeholder")
        self.document_sentence_placeholder = tf.placeholder(tf.int32, shape=(None, FLAGS.max_document_size + 1),
                                                            name="document_sentence_placeholder")
        self.span_placeholder = tf.placeholder(tf.int32, shape=(None, 3),
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
        if 'd_s' in data_batch and data_batch['d_s'] is not None:
            feed_dict[self.document_sentence_placeholder] = data_batch['d_s']
        if 'd_n_s' in data_batch and data_batch['d_n_s'] is not None:
            feed_dict[self.document_n_sentence_placeholder] = data_batch['d_n_s']
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
        all_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, trainable=False)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)
        return question_embeddings, document_embeddings

    def summarize(self, x, D, q_sen):
        sentences = []
        sen_len = []
        sen_rep = []
        n_sentence = 0  # number of sentences in document

        for sen in range(FLAGS.max_document_size):
            idx_from = tf.document_sentence_placeholder(x)  # sentence begin word index in document
            idx_to = tf.document_sentence_placeholder(x + 1)  # sentence end word index in document
            if idx_to < 0: # This is no longer a sentence
                break

            sentences.append(D[x, idx_from:idx_to, :])
            sen_len.append(idx_to + 1 - idx_from)
            n_sentence += 1
            
            # Sentence-level representation
            rep = tf.reduce_max(sentences[-1], axis=0) if FLAGS.model.lower() == "max" \
                                                       else tf.reduce_mean(sentences[-1], axis=0)
            assert_shape(rep, "rep", [FLAGS.state_size])
            sen_rep.append(rep)
        
        # Normalized sentence-level representation
        sen_rep = tf.stack(sen_rep, axis=0)
        assert_shape(sen_rep, "sen_rep", [n_sentence, FLAGS.state_size])
        sen_rep /= tf.sqrt(tf.reduce_sum(tf.square(sen_rep), axis=1, keep_dims=True))
        assert_shape(sen_rep, "sen_rep", [n_sentence, FLAGS.state_size])

        # Similarity between the question rep and each sentence rep
        sen_sim = tf.matmul(q_sen[x, :], tf.transpose(sen_rep))

        # Reorder sentence in document, then truncate doc to the maximum summary length
        (_, sen_sim_idx) = tf.nn.top_k(sen_sim, k=n_sentence)
        sen_sorted = [sentences[i] for i in sen_sim_idx]
        # if idx_from < FLAGS.max_document_size:
        sen_sorted.append(D[x, idx_from:, :])  # Non-empty if padding is in document
        D_summary = tf.concat(0, sen_sorted)
        assert_shape(D_summary, "D_summary", [FLAGS.max_document_size, FLAGS.state_size])
        D_summary = D_summary[:FLAGS.max_summary_size, :]
        assert_shape(D_summary, "D_summary", [FLAGS.max_summary_size, FLAGS.state_size])

        # Update answer span in the to the index in summary
        # NOTE: If the answer is not located in a sentence in the summary, then the 
        #       eventual calculatd span would be larger than FLAGS.max_summary_size and 
        #       produce NaN when calculating cross entropy. This is remedied by ignoring 
        #       NaN values when averaging the loss.
        ans_sen_order = np.argmax(sen_sim_idx == self.span_placeholder[x, 2])
        len_b4_document = sum(sen_len[:ans_sen_order])
        len_b4_summary = sum(sen_len[sen_sim_idx[i]] for i in range(ans_sen_order))
        self.span_placeholder[x, 0] += len_b4_summary - len_b4_document
        self.span_placeholder[x, 1] += len_b4_summary - len_b4_document

        return D_summary

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
        
        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])
        assert_shape(D, "D", [None, FLAGS.max_document_size, FLAGS.state_size])

        if FLAGS.max_summary_size != FLAGS.max_document_size:
            if FLAGS.model.lower() == "max":
                q_sen = tf.reduce_max(Q, axis=1)
            elif FLAGS.model.lower() == "mean":
                q_sen = tf.reduce_mean(Q, axis=1)
            else:
                q_sen = None

            assert_shape(q_sen, "Q_sen", [None, FLAGS.state_size])
            D = tf.map_fn(lambda x: self.summarize(x, D, q_sen), range(len(D.shape[0])), dtype=tf.float32)

        # Non-linear projection layer on top of the question encoding.
        with tf.variable_scope("Q-TANH"):
            W_q = tf.get_variable("W_q", shape=(FLAGS.state_size, FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            b_q = tf.get_variable("b_q", shape=(FLAGS.state_size),
                                  dtype=tf.float32, initializer=tf.constant_initializer(0.))
            # Q = tf.tanh(tf.einsum('ijk,kl->ijl', Q, W_q) + b_q)
            Q = tf.scan(lambda a, x: tf.matmul(x, W_q), Q)
            Q = tf.tanh(Q +  b_q)


        assert_shape(Q, "Q", [None, FLAGS.max_question_size, FLAGS.state_size])
        assert_shape(D, "D", [None, FLAGS.max_document_size, FLAGS.state_size])
        return (Q, D)

    ## ==============================
    ## COATTENTION ENCODER
    def encode(self, preprocessing, debug_shape=False):
        Q = preprocessing[0]
        D = preprocessing[1]

        # Affinity matrix.
        L = tf.batch_matmul(Q, tf.transpose(D, [0, 2, 1]))
        assert_shape(L, "L", [None, FLAGS.max_question_size, FLAGS.max_document_size])

        # Normalize with respect to question/document.
        A_q = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
        assert_shape(A_q, "A_q", [None, FLAGS.max_question_size, FLAGS.max_document_size])
        A_d = tf.map_fn(lambda x: tf.nn.softmax(x), tf.transpose(L, [0, 2, 1]), dtype=tf.float32)
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
            cell_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            (U, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, coatt, dtype=tf.float32, \
                sequence_length=self.document_seq_placeholder)
            U = tf.concat(2,U)
        
        assert_shape(U, "U", [None, FLAGS.max_document_size, 2 * FLAGS.state_size])
        return U

    ## ==============================
    ## FEED FORWARD DECODER
    def feed_forward_decode(self, coattention, debug_shape=False):
        Hr = coattention
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

            answer_pointer_rep = (betas, pred)

        return answer_pointer_rep

    
    ## ==============================
    ## ANSWER POINTER DECODER
    def answer_pointer(self, coattention, debug_shape=False):
        H_r = coattention
        assert_shape(H_r, "H_r", [None, FLAGS.max_document_size, 2 * FLAGS.state_size])

        with tf.variable_scope("ANSWER_POINTER"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=FLAGS.state_size, state_is_tuple=True)
            ha = cell.zero_state(FLAGS.batch_size, tf.float32)
            assert_shape(ha[1], "ha[1]", [FLAGS.batch_size, FLAGS.state_size])
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

                VH_r = tf.einsum('ijk,kl->ijl', H_r, V)
                assert_shape(VH_r, "VH_r", [FLAGS.batch_size, FLAGS.max_document_size, FLAGS.state_size])
                W_aH_ab_a = tf.matmul(ha[1], W_a) + b_a
                assert_shape(W_aH_ab_a, "W_aH_ab_a", [FLAGS.batch_size, FLAGS.state_size])
                W_aH_ab_a = tf.expand_dims(W_aH_ab_a, axis=1)
                F_k = tf.nn.tanh(VH_r + tf.tile(W_aH_ab_a, [1, FLAGS.max_document_size, 1]))
                F_k = tf.transpose(F_k, perm=[0, 2, 1])
                assert_shape(F_k, "F_k", [FLAGS.batch_size, FLAGS.state_size, FLAGS.max_document_size])
                
                v_tF_k = tf.einsum('ij,kjl->kil', v, F_k)
                assert_shape(v_tF_k, "v_tF_k", [FLAGS.batch_size, 1, FLAGS.max_document_size])
                beta_k = tf.nn.softmax(v_tF_k + tf.tile(c, [1, FLAGS.max_document_size]))
                assert_shape(beta_k, "beta_k", [FLAGS.batch_size, 1, FLAGS.max_document_size])

                H_rbeta_k = tf.squeeze(tf.batch_matmul(beta_k, H_r), squeeze_dims=1)
                assert_shape(H_rbeta_k, "H_rbeta_k", [FLAGS.batch_size, 2 * FLAGS.state_size])

                beta.append(beta_k)
                (_, ha) = cell(H_rbeta_k, ha)

            beta = tf.concat(1,beta)
            assert_shape(beta, "beta", [FLAGS.batch_size, 2, FLAGS.max_document_size])
        return (beta, tf.argmax(beta, axis=2))

    def loss(self, decoded, debug_shape=False):
        beta = decoded[0]
        y = self.span_placeholder
        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 0, :], y[:, 0]))
        L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(beta[:, 1, :], y[:, 1]))
        return (L1 + L2) / 2.0

    def add_training_op(self, loss, debug_shape=False):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        gradients = optimizer.compute_gradients(loss)
        (grad, var) = zip(*gradients)
        (grad, _) = tf.clip_by_global_norm(grad, FLAGS.max_gradient_norm)

        grad_norm = []
        logger.info("----------all trainable variables picked for grad norm------------------")
        for i,v in enumerate(var):
            logger.info(v.name)
            grad_norm.append(tf.global_norm([grad[i]]))
        grad_norm = tf.pack(grad_norm)

        train_op = optimizer.apply_gradients(zip(grad, var))
        return (train_op, grad_norm, loss)

    def build(self, debug_shape):
        self.add_placeholders()
        self.preprocessed = self.preprocessing(debug_shape)
        self.encoded = self.encode(self.preprocessed, debug_shape)
        # self.decoded = self.answer_pointer(self.encoded, debug_shape)
        self.decoded = self.feed_forward_decode(self.encoded, debug_shape)
        self.lost = self.loss(self.decoded, debug_shape)
        self.train_op = self.add_training_op(self.lost, debug_shape)

    def debug_shape(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        debug_output = sess.run(
            fetches = util.tuple_to_list(*self.decoded),
            feed_dict=feed
        )
        # logger.info("grads: {}".format(train_op_output[1]))
        # logger.info("loss: {}".format(train_op_output[2]))
        # # logger.info("pred: {}".format(train_op_output[4]))
        # logger.info(debug_output)
        for i, tensor in enumerate(self.decoded):
            if tensor.name.startswith("debug_"):
                logger.debug("Shape of {} == {}".format(tensor.name[6:], debug_output[i]))

    def predict_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)
        answer_pointer_rep = sess.run(
            fetches = util.tuple_to_list(*self.decoded),
            feed_dict=feed
        )
        pred = du.get_answer_from_span(answer_pointer_rep[1])
        return pred

    def train_on_batch(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op = sess.run(
            fetches = util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )

        grad_norm = train_op[1]
        loss = train_op[2]
        # pred = du.get_answer_from_span(train_op[4])

        return grad_norm, loss
