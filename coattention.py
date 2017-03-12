import logging

import tensorflow as tf

import util
import qa_data_util as du

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def assertion(var_name, expected):
    shape = eval(var_name).get_shape().as_list()
    assert shape == expected \
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

        assertion("question_embeddings", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.state_size])
        assertion("document_embeddings", [FLAGS.batch_size, FLAGS.max_document_size + 1, FLAGS.state_size])

        return question_embeddings, document_embeddings

    ## ==============================
    ## DOCUMENT AND QUESTION ENCODER
    def add_preprocessing_op(self, debug_shape=False):
        (Q_embed, D_embed) = self.add_embedding()

        # Encoding question and document.
        with tf.variable_scope("QD-ENCODE"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)
            (Q, _) = tf.nn.dynamic_rnn(cell, Q_embed)
            tf.get_variable_scope().reuse_variables()
            (D, _) = tf.nn.dynamic_rnn(cell, D_embed)
        
        assertion("Q", [FLAGS.batch_size, FLAGS.max_question_size, FLAGS.state_size])
        assertion("D", [FLAGS.batch_size, FLAGS.max_document_size, FLAGS.state_size])

        # Add sentinel to the end of document.
        Q = tf.concat_v2([Q, tf.zeros([FLAGS.batch_size, 1, FLAGS.state_size])], 1)
        D = tf.concat_v2([D, tf.zeros([FLAGS.batch_size, 1, FLAGS.state_size])], 1)

        # Non-linear projection layer on top of the question encoding.
        with tf.variable_scope("Q-TANH"):
            Wq = tf.get_variable("Wq", shape=(FLAGS.state_size, FLAGS.state_size)
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bq = tf.get_variable("bq", shape=(FLAGS.state_size)
                                 dtype=tf.float32, initializer=tf.constant_initializer(0.))
            Q = tf.tanh(tf.matmul(Q, Wq) + bq)

        assertion("Q", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.state_size])
        assertion("D", [FLAGS.batch_size, FLAGS.max_document_size + 1, FLAGS.state_size])
        return (Q, D)

    ## ==============================
    ## COATTENTION ENCODER
    def add_coattention_op(self, preprocessing, debug_shape=False):
        Q = preprocessing[0]
        D = preprocessing[1]

        # Affinity matrix.
        L = tf.matmul(Q, tf.transpose(D))
        assertion("L", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.max_document_size + 1])

        # Normalize with respect to question/document.
        Aq = tf.nn.softmax(L, dim=2)
        assertion("Aq", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.max_document_size + 1])
        Ad = tf.transpost(tf.nn.softmax(L, dim=1))
        assertion("Ad", [FLAGS.batch_size, FLAGS.max_document_size + 1, FLAGS.max_question_size + 1])

        # Attention of the document w.r.t question.
        Cq = tf.matmul(Aq, D)
        assertion("Cq", [FLAGS.batch_size, FLAGS.max_question_size + 1, FLAGS.state_size])

        # Attention of previous attention w.r.t document, concatenated with attention of
        # question w.r.t. document.
        Cd = tf.concat_v2([tf.matmul(Ad, Q), tf.matmul(Ad, Cq)], 2)
        assertion("Cd", [FLAGS.batch_size, FLAGS.max_document_size + 1, 2 * FLAGS.state_size])

        # Fusion of temporal information to the coattention context
        coatt = tf.concat_v2([D, Cd], 2)
        assertion("coatt", [FLAGS.batch_size, FLAGS.max_document_size + 1, 3 * FLAGS.state_size])

        with tf.variable_scope("COATTENTION"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.state_size)
            (U, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, coatt, dtype=tf.float32)
            U = tf. concat_v2(U, 2)
        
        assertion("U", [FLAGS.batch_size, FLAGS.max_document_size + 1, 2 * FLAGS.state_size])


    ## ==============================
    ## DYNAMIC POINTING DECODER
    def add_answer_pointer_op(self, match_lstm_rep, debug_shape=False):
        Hr = match_lstm_rep[0]

        with tf.variable_scope("ANSWER_POINTER"):
            V =tf.get_variable(name='V',
                                 shape = [2*FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()
                                 )

            W_a =tf.get_variable(name='W_a',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()
                                 )

            b_a =tf.get_variable(name='b_a',
                                 shape = [FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            v =tf.get_variable(name='v',
                                 shape = [FLAGS.state_size, 1],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            c =tf.get_variable(name='c',
                                 shape = (1,),
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)

            ha = cell.zero_state(FLAGS.batch_size, tf.float32)
            betas = []
            for k in range(2):
                if k > 0:
                    tf.get_variable_scope().reuse_variables()
                V_Hr = tf.einsum('ijk,kl->ijl', Hr, V)
                Wa_Ha = tf.matmul(ha[1], W_a)
                Fk = Wa_Ha + b_a
                Fk = tf.reshape(
                    tensor=tf.tile(Fk, [1,FLAGS.max_document_size]),
                    shape=[FLAGS.batch_size, FLAGS.max_document_size, FLAGS.state_size]
                )
                Fk = tf.nn.tanh(Fk + V_Hr)

                vt_Fk = tf.reshape(tf.einsum('ijk,kl->ijl', Fk, v),[FLAGS.batch_size, FLAGS.max_document_size])

                betak = tf.nn.softmax(vt_Fk + tf.tile(c, [FLAGS.max_document_size]))
                betak_ = tf.reshape(betak,[FLAGS.batch_size, 1,FLAGS.max_document_size])

                Hr_betak = tf.einsum('ijk,ikl->ijl', betak_, Hr)
                Hr_betak = tf.reshape(Hr_betak, [FLAGS.batch_size, 2*FLAGS.state_size])


                betas.append(betak)
                _, ha = cell(Hr_betak, ha)

            betas = tf.pack(betas, 1)

        pred = tf.argmax(betas,2)

        answer_pointer_rep = (betas, pred)
        if debug_shape:
            return answer_pointer_rep+(
                tf.shape(V_Hr,name="debug_APL_V_Hr"),
                tf.shape(Fk,name="debug_APL_Fk"),
                tf.shape(vt_Fk,name="debug_APL_vt_fk"),
                tf.shape(betak,name="debug_APL_betak"),
                tf.shape(Hr_betak,name="debug_APL_Hr_betak"),
                tf.shape(betas,name="debug_APL_betas"),
                tf.shape(pred,name="debug_APL_pred"),
            ) + match_lstm_rep

        return  answer_pointer_rep + match_lstm_rep

    def add_loss_op(self, answer_pointer_rep, debug_shape=False):
        betas = answer_pointer_rep[0]
        y = self.span_placeholder
        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas[:,0,:], y[:,0]))
        L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas[:,1,:], y[:,1]))
        return ((L1+L2)/2.0,) + answer_pointer_rep

    def add_training_op(self, loss, debug_shape=False):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss[0])

        return (train_op,) + loss

    def build(self, debug_shape):
        self.add_placeholders()
        self.preprocessing_rep = self.add_preprocessing_op(debug_shape)
        self.match_lstm_rep = self.add_coattention_op(self.preprocessing_rep, debug_shape)
        self.answer_pointer_rep = self.add_answer_pointer_op(self.match_lstm_rep, debug_shape)
        self.loss = self.add_loss_op(self.answer_pointer_rep, debug_shape)
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
        answer_pointer_rep = sess.run(
            fetches = util.tuple_to_list(*self.answer_pointer_rep),
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

        loss = train_op[1]
        pred = du.get_answer_from_span(train_op[3])

        return loss, pred
