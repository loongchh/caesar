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

        self.exploded_span_placeholder = tf.placeholder(tf.int32,
                                                        shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                        name="exploded_span_placeholder")

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

        if data_batch['s_e'] is not None:
            feed_dict[self.exploded_span_placeholder] = data_batch['s_e']

        return feed_dict

    def add_embedding(self):
        all_embeddings = tf.cast(
            tf.get_variable("embeddings", initializer=self.pretrained_embeddings, dtype=tf.float64),
            tf.float32)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)

        return question_embeddings, document_embeddings

    def add_encoder_op(self, debug_shape=False):

        ###################################
        #####  LSTM preprocessing Layer ###
        ####################################

        q_input,d_input = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("Q_LSTM"):

            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)

            initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

            (output, _) = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=q_input,
                                            initial_state=initial_state,
                                            sequence_length=self.question_seq_placeholder
                                            )
            H_Q = output

        with tf.variable_scope("P_LSTM"):

            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)

            initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

            (output, _) = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=tf.transpose(d_input,perm=[1,0,2]),
                                            initial_state=initial_state,
                                            sequence_length=self.question_seq_placeholder,
                                            time_major=True
                                            )
            H_P = output

        representations = (H_Q,H_P)

        if debug_shape:
            return representations + (
                tf.shape(d_input,name="debug_d_input"),
                tf.shape(q_input,name="debug_q_input"),
                tf.shape(H_Q,name="debug_H_Q"),
                tf.shape(H_P,name="debug_H_p")
            )
        return representations

    def add_decoder_op(self, encoded_representation, debug_shape=False):
        H_Q = encoded_representation[0]
        H_P = tf.unpack(encoded_representation[1])

        ####################################
        #####  Match LSTM Layer #########
        ####################################

        with tf.variable_scope("Match_LSTM"):
            W_q =tf.get_variable(name='W_q',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )
            W_p =tf.get_variable(name='W_p',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            W_r =tf.get_variable(name='W_r',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            b_p =tf.get_variable(name='b_p',
                                 shape = [FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            w =tf.get_variable(name='w',
                                 shape = [FLAGS.state_size, 1],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            b =tf.get_variable(name='b',
                                 shape = (1,),
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )


            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)

            hr = cell.zero_state(FLAGS.batch_size, tf.float32)
            Hr = []
            for i, H_Pi in enumerate(H_P):
                if i>0:
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

            Hr = tf.concat(1, [tf.zeros(shape=[10,1,FLAGS.state_size]), Hr] )


        ####################################
        ##### Answer Pointer Layer #########
        ####################################

        with tf.variable_scope("ANSWER_POINTER"):
            V =tf.get_variable(name='V',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            W_a =tf.get_variable(name='W_a',
                                 shape = [FLAGS.state_size, FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            b_a =tf.get_variable(name='b_a',
                                 shape = [FLAGS.state_size],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            v =tf.get_variable(name='v',
                                 shape = [FLAGS.state_size, 1],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )

            c =tf.get_variable(name='c',
                                 shape = (1,),
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=1e-4)
                                 )


            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0, state_is_tuple=True)

            ha = cell.zero_state(FLAGS.batch_size, tf.float32)
            betas = []
            for k in range(FLAGS.max_document_size):
                if k > 0:
                    tf.get_variable_scope().reuse_variables()
                V_Hr = tf.einsum('ijk,kl->ijl', Hr, V)
                Wa_Ha = tf.matmul(ha[1], W_a)
                Fk = Wa_Ha + b_p
                Fk = tf.reshape(
                    tensor=tf.tile(Fk, [1,FLAGS.max_document_size+1]),
                    shape=[FLAGS.batch_size, FLAGS.max_document_size+1, FLAGS.state_size]
                )
                Fk = tf.nn.tanh(Fk + V_Hr)

                vt_Fk = tf.reshape(tf.einsum('ijk,kl->ijl', Fk, v),[FLAGS.batch_size, FLAGS.max_document_size+1])

                betak = tf.nn.softmax(vt_Fk + tf.tile(c, [FLAGS.max_document_size+1]))
                betak_ = tf.reshape(betak,[FLAGS.batch_size, 1,FLAGS.max_document_size+1])
            #
                Hr_betak = tf.einsum('ijk,ikl->ijl', betak_, Hr)
                Hr_betak = tf.reshape(Hr_betak, [FLAGS.batch_size, FLAGS.state_size])

            #
                betas.append(betak)
                _, ha = cell(Hr_betak, ha)

            betas = tf.pack(betas, 1)

            # betas = tf.reshape(tf.pack(betas, 1), [FLAGS.batch_size, FLAGS.max_document_size*FLAGS.max_document_size])

        pred = tf.constant(1,shape=[FLAGS.batch_size, 2], dtype=tf.int32 )

        debug_info = (
            tf.shape(H_P,name="debug_HP"),
            tf.shape(H_Q,name="debug_HQ"),
            tf.shape(H_P[0],name="debug_HP0"),
            tf.shape(Wq_HQ,name="debug_Wq_HQ"),
            tf.shape(Wp_HPi,name="debug_Wp_HPi"),
            tf.shape(Wr_Hr,name="debug_Wr_Hr"),
            tf.shape(Gi,name="debug_Gi"),
            tf.shape(wt_Gi,name="debug_wt_Gi"),
            tf.shape(alphai,name="debug_alphai"),
            tf.shape(HQ_alphai,name="debug_HQ_alphai"),
            tf.shape(zi,name="debug_zi"),
            tf.shape(Hr,name="debug_Hr"),
            tf.shape(V_Hr,name="debug_V_Hr"),
            tf.shape(Fk,name="debug_Fk"),
            tf.shape(vt_Fk,name="debug_vt_fk"),
            tf.shape(betak,name="debug_betak"),
            tf.shape(Hr_betak,name="debug_Hr_betak"),
            tf.shape(betas,name="debug_betas"),
        )




        if debug_shape:
            return (betas, pred) + debug_info #+encoded_representation
        return  (betas, pred)


    def add_loss_op(self, decoded_representation, debug_shape=False):
        betas = decoded_representation[0]
        pred = decoded_representation[1]
        y = self.exploded_span_placeholder
        # diff = y[:,1] - y[:,0]
        L1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas, y))
        # L2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(betas[:,diff,: ], y[:,1]))
        return (L1, pred)

    def add_training_op(self, loss, debug_shape=False):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss[0])

        return (train_op,) + loss
