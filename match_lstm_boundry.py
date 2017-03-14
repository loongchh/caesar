import logging

import tensorflow as tf

import util
import qa_data_util as du
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class MatchLstmBoundryModel():

    def __init__(self, embeddings, debug_shape=False):
        self.pretrained_embeddings = embeddings
        self.build(debug_shape)


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

        self.answer_placeholder = tf.placeholder(tf.int32,
                                                        shape=(FLAGS.batch_size, FLAGS.max_answer_size),
                                                        name="answer_placeholder")

        self.answer_mask_placeholder = tf.placeholder(tf.bool,
                                                        shape=(FLAGS.batch_size, FLAGS.max_answer_size),
                                                        name="answer_mask_placeholder")
        self.answer_seq_placeholder = tf.placeholder(tf.int32,
                                                        shape=(FLAGS.batch_size, ),
                                                        name="answer_seq_placeholder")

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

        if data_batch['a'] is not None:
            feed_dict[self.answer_placeholder] = data_batch['a']

        if data_batch['a_m'] is not None:
            feed_dict[self.answer_mask_placeholder] = data_batch['a_m']

        if data_batch['a_s'] is not None:
            feed_dict[self.answer_seq_placeholder] = data_batch['a_s']

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.get_variable("embeddings", initializer=self.pretrained_embeddings, dtype=tf.float32, trainable=True)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)

        return question_embeddings, document_embeddings



    ###################################
    #####  LSTM preprocessing Layer ###
    ####################################
    def add_preprocessing_op(self, debug_shape=False):

        q_input,d_input = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("Q_LSTM"):

            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)

            initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

            (output, _) = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=q_input,
                                            initial_state=initial_state,
                                            sequence_length=self.question_seq_placeholder
                                            )
            H_Q = output

        with tf.variable_scope("P_LSTM"):

            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)

            initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

            (output, _) = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=tf.transpose(d_input,perm=[1,0,2]),
                                            initial_state=initial_state,
                                            sequence_length=self.question_seq_placeholder,
                                            time_major=True
                                            )
            H_P = output

        preprocessing_rep = (H_Q,H_P)

        if debug_shape:
            return preprocessing_rep + (
                tf.shape(d_input,name="debug_PLL_d_input"),
                tf.shape(q_input,name="debug_PLL_q_input"),
                tf.shape(H_Q,name="debug_PLL_H_Q"),
                tf.shape(H_P,name="debug_PLL_H_p")
            )
        return preprocessing_rep

    ####################################
    #####  Match LSTM Layer #########
    ####################################
    def add_match_lstm_op(self, preprocessing_rep, debug_shape=False):
        H_Q = preprocessing_rep[0]
        H_P = tf.unpack(preprocessing_rep[1])
        fwd = self.match_lstm_direction_op(H_P, H_Q, direction='fwd',debug_shape=debug_shape)
        rev = self.match_lstm_direction_op(H_P, H_Q, direction='rev',debug_shape=debug_shape)

        Hr = tf.concat(2, [fwd[0], rev[0]])
        match_lstm_rep = (Hr,)
        if debug_shape:
            return match_lstm_rep + (tf.shape(Hr,name="debug_MLL_Hr"),) + fwd + rev + preprocessing_rep
        return match_lstm_rep + preprocessing_rep

    #  Match LSTM Forward/Bacward Layer #########
    def match_lstm_direction_op(self, H_P, H_Q, direction, debug_shape=False):
        if direction == "rev":
            tf.reverse(H_P, [True, False, False])
        with tf.variable_scope("Match_LSTM_{}".format(direction)):
            W_q =tf.get_variable(name='W_q',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()
                                 )
            W_p =tf.get_variable(name='W_p',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()
                                 )

            W_r =tf.get_variable(name='W_r',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer()
                                 )

            b_p =tf.get_variable(name='b_p',
                                 shape = [FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            w =tf.get_variable(name='w',
                                 shape = [FLAGS.state_size, 1],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )

            b =tf.get_variable(name='b',
                                 shape = (1,),
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0)
                                 )


            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size)

            hr = cell.zero_state(FLAGS.batch_size, tf.float32)
            Hr = []
            for i, H_Pi in enumerate(H_P):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                Wq_HQ = tf.einsum('ijk,kl->ijl', H_Q, W_q)
                Wp_HPi = tf.matmul(H_P[i], W_p)
                Wr_Hr = tf.matmul(hr[1], W_r)
                Gi = Wp_HPi + Wr_Hr + b_p
                Gi = tf.reshape(
                    tensor=tf.tile(Gi, [1,FLAGS.max_question_size]),
                    shape=[FLAGS.batch_size, FLAGS.max_question_size, FLAGS.state_size]
                )
                Gi = tf.nn.tanh(Gi + Wq_HQ)
                wt_Gi = tf.reshape(tf.einsum('ijk,kl->ijl', Gi, w),[FLAGS.batch_size, FLAGS.max_question_size])

                alphai = tf.nn.softmax(wt_Gi + tf.tile(b, [FLAGS.max_question_size]))
                alphai = tf.reshape(alphai,[FLAGS.batch_size, 1,FLAGS.max_question_size])

                HQ_alphai = tf.einsum('ijk,ikl->ijl', alphai, H_Q)
                HQ_alphai = tf.reshape(HQ_alphai, [FLAGS.batch_size, FLAGS.state_size])

                zi = tf.concat(1, [H_P[i], HQ_alphai])

                _, hr = cell(zi, hr)
                Hr.append(hr[1])

            Hr = tf.pack(Hr,1)

        match_lstm_rep = (Hr,)
        if debug_shape:
            return match_lstm_rep + (
                tf.shape(H_P,name="debug_MLL_{}_HP".format(direction)),
                tf.shape(H_Q,name="debug_MLL_{}_HQ".format(direction)),
                tf.shape(H_P[0],name="debug_MLL_{}_HP0".format(direction)),
                tf.shape(Wq_HQ,name="debug_MLL_{}_Wq_HQ".format(direction)),
                tf.shape(Wp_HPi,name="debug_MLL_{}_Wp_HPi".format(direction)),
                tf.shape(Wr_Hr,name="debug_MLL_{}_Wr_Hr".format(direction)),
                tf.shape(Gi,name="debug_MLL_{}_Gi".format(direction)),
                tf.shape(wt_Gi,name="debug_MLL_{}_wt_Gi".format(direction)),
                tf.shape(alphai,name="debug_MLL_{}_alphai".format(direction)),
                tf.shape(HQ_alphai,name="debug_MLL_{}_HQ_alphai".format(direction)),
                tf.shape(zi,name="debug_MLL_{}_zi".format(direction)),
                tf.shape(Hr,name="debug_MLL_{}_Hr".format(direction)),
            )

        return match_lstm_rep

    ####################################
    ##### Answer Pointer Layer #########
    ####################################
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


        return answer_pointer_rep + match_lstm_rep

    def add_loss_op(self, answer_pointer_rep, debug_shape=False):
        betas = answer_pointer_rep[0]
        y = self.span_placeholder
        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas[:,0,:], y[:,0]))
        L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas[:,1,:], y[:,1]))
        return ((L1+L2)/2.0,) + answer_pointer_rep

    def add_training_op(self, loss, debug_shape=False):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        gradients = optimizer.compute_gradients(loss[0])
        (grad, var) = zip(*gradients)

        (grad, _) = tf.clip_by_global_norm(grad, 15.0)

        grad_norm = []
        logger.info("----------all trainable variables picked for grad norm------------------")
        for i,v in enumerate(var):

            logger.info(v.name)
            grad_norm.append(tf.global_norm([grad[i]]))
        grad_norm = tf.pack(grad_norm)
        train_op = optimizer.apply_gradients(zip(grad, var))

        return (train_op,grad_norm) + loss + (tf.shape(grad_norm, name="debug_norm_shape"),)

    def build(self, debug_shape):
        self.add_placeholders()
        self.preprocessing_rep = self.add_preprocessing_op(debug_shape)
        self.match_lstm_rep = self.add_match_lstm_op(self.preprocessing_rep, debug_shape)
        self.answer_pointer_rep = self.add_answer_pointer_op(self.match_lstm_rep, debug_shape)
        self.loss = self.add_loss_op(self.answer_pointer_rep, debug_shape)
        self.train_op = self.add_training_op(self.loss, debug_shape)

    def debug_shape(self, sess, data_batch):
        feed = self.create_feed_dict(data_batch)

        train_op_output = sess.run(
            fetches = util.tuple_to_list(*self.train_op),
            feed_dict=feed
        )
        logger.info("grads: {}".format(train_op_output[1]))
        logger.info("loss: {}".format(train_op_output[2]))
        logger.info("pred: {}".format(train_op_output[4]))

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

        grad_norm = train_op[1]
        loss = train_op[2]
        pred = du.get_answer_from_span(train_op[4])

        return grad_norm, loss, pred
