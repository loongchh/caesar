import logging, time
import tensorflow as tf
import numpy as np
from q_a_model import QAModel
import qa_data_util as du
import parse_args
FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class CoattentionModel(QAModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)

        self.question_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, FLAGS.max_question_size),
                                                name="question_placeholder")
        self.document_placeholder = tf.placeholder(tf.int32,
                                                shape=(FLAGS.batch_size, FLAGS.max_document_size),
                                                name="document_placeholder")
        self.span_placeholder = tf.placeholder(tf.int32,
                                                 shape=(FLAGS.batch_size, 2),
                                                 name="span_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  name="dropout_placeholder")

        ### END YOUR CODE

    def create_feed_dict(self, question_batch, document_batch, span_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)

        feed_dict = {
            self.question_placeholder: question_batch,
            self.document_placeholder: document_batch
        }

        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        if span_batch is not None:
            feed_dict[self.span_placeholder] = span_batch

        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        all_embeddings = tf.cast(
            tf.get_variable("all_embeddings", initializer=self.pretrained_embeddings, dtype=tf.float64),
            tf.float32)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)


        ### END YOUR CODE
        return question_embeddings, document_embeddings

    def add_encoder_op(self):
        q_input,d_input = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("BLSTM"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)

            initial_state_fw = cell_fw.zero_state(FLAGS.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(FLAGS.batch_size, tf.float32)

            ## calculate document representation

            d = tf.transpose(d_input, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            d = tf.reshape(d, [-1, FLAGS.vocab_dim])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            d = tf.split(0, FLAGS.max_document_size, d)
            output, _, _ = tf.nn.bidirectional_rnn(cell_fw=cell_fw,
                                                   cell_bw=cell_bw,
                                                   inputs=d,
                                                   initial_state_fw=initial_state_fw,
                                                   initial_state_bw=initial_state_bw,
                                                   scope="BLSTM",
                                                   )
            d_representation = tf.transpose(output, perm=[1,0,2])
            d_representation = tf.nn.dropout(d_representation, dropout_rate)

            """
            resuse the same lstm for questions respresetation
            """
            q = tf.transpose(q_input, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            q = tf.reshape(q, [-1, FLAGS.vocab_dim])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            q = tf.split(0, FLAGS.max_question_size, q)
            tf.get_variable_scope().reuse_variables()
            output, _, _ = tf.nn.bidirectional_rnn(cell_fw=cell_fw,
                                                   cell_bw=cell_bw,
                                                   inputs=q,
                                                   initial_state_fw=initial_state_fw,
                                                   initial_state_bw=initial_state_bw,
                                                   scope="BLSTM")
            q_representation = tf.transpose(output, perm=[1,0,2])
            q_representation = tf.nn.dropout(q_representation, dropout_rate)

        Wq = tf.get_variable(
            initializer=tf.contrib.layers.xavier_initializer(),
            shape=(2 * FLAGS.state_size, 2 * FLAGS.state_size),
            name="Wq",
            dtype=tf.float32
        )
        bq = tf.Variable(
            tf.zeros_initializer(
                shape=( 2 * FLAGS.state_size,),
                dtype=tf.float32
            ),
            name="bq"
        )
        # q_representation = tf.nn.tanh(tf.einsum('ijk,kk->ijk',q_representation, Wq) + bq)


        assert d_representation.get_shape().as_list() == [FLAGS.batch_size, FLAGS.max_document_size, 2 * FLAGS.state_size], \
            "document representation are not of the right shape. Expected {}, got {}".format([FLAGS.batch_size, FLAGS.max_document_size, 2 * FLAGS.state_size], d_representation.get_shape().as_list())

        assert q_representation.get_shape().as_list() == [FLAGS.batch_size, FLAGS.max_question_size, 2 * FLAGS.state_size], \
            "questions representation are not of the right shape. Expected {}, got {}".format([FLAGS.batch_size, FLAGS.max_question_size, 2 * FLAGS.state_size], q_representation.get_shape().as_list())

        q_representation_transpose = tf.transpose(q_representation, perm=[0,2,1])
        d_representation_transpose = tf.transpose(d_representation, perm=[0,2,1])
        # return tf.shape(q_representation_transpose),tf.shape(d_representation_transpose)
        # exit()

        ###  Weirdo Tensorflow
        # L = tf.batch_matmul( tf.transpose(d_representation,perm=[0,2,1]), q_representation)

        L = tf.batch_matmul(d_representation,q_representation_transpose)

        Aq = tf.nn.softmax(L) # batch_size * max_document_size * max_question_size
        Ad = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]))   # batch_size *  max_question_size * max_document_size *

        Cq = tf.batch_matmul(tf.transpose(Aq,[0,2,1]), d_representation)  # batch_size * 300 * 60
        QCq = tf.concat(2, [q_representation, Cq])
        Cd = tf.batch_matmul(tf.transpose(Ad, perm=[0,2,1]), QCq)
        DCd = tf.concat(2, [d_representation, Cd])


        with tf.variable_scope("BLSTM2"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)

            initial_state_fw = cell_fw.zero_state(FLAGS.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(FLAGS.batch_size, tf.float32)

            ## calculate document representation

            DCd_ = tf.transpose(DCd, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            DCd_ = tf.reshape(DCd_, [-1, 6 * FLAGS.state_size])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            DCd_ = tf.split(0, FLAGS.max_document_size, DCd_)
            output, _, _ = tf.nn.bidirectional_rnn(cell_fw=cell_fw,
                                                   cell_bw=cell_bw,
                                                   inputs=DCd_,
                                                   initial_state_fw=initial_state_fw,
                                                   initial_state_bw=initial_state_bw,
                                                   scope="BLSTM2",
                                                   )
            encoded_representation = tf.transpose(output, perm=[1,0,2], name="encoded_representation")

        # return (
        #     encoded_representation,
        #     tf.shape(d_input,name="debug_d"),
        #     tf.shape(q_input,name="debug_q"),
        #     tf.shape(d_representation,name="debug_d_representation"),
        #     tf.shape(q_representation, name="debug_q_representation"),
        #     tf.shape(d_representation_transpose,name="debug_d_representation_transpose"),
        #     tf.shape(q_representation_transpose,name="debug_q_representation_transpose" ),
        #     tf.shape(L, name="debug_L"),
        #     tf.shape(Aq, name="debug_Aq"),
        #     tf.shape(Ad, name="debug_Ad"),
        #     tf.shape(Cq, name="debug_Cq"),
        #     tf.shape(QCq, name="debug_QCq"),
        #     tf.shape(Cd, name="debug_Cd"),
        #     tf.shape(DCd, name="debug_DCd"),
        #     tf.shape(encoded_representation, name="debug_encoded_representation")
        # )
        return encoded_representation,

    def add_decoder_op(self, encoded_representation):
        assert isinstance(encoded_representation, tuple)
        batch_range = tf.range(FLAGS.batch_size)
        zeros = tf.zeros(FLAGS.batch_size, dtype=tf.int32)
        ones = tf.ones(FLAGS.batch_size, dtype=tf.int32)
        U = encoded_representation[0]

        loss = tf.zeros([1])

        with tf.variable_scope("LSTM"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)

            h = cell.zero_state(FLAGS.batch_size, tf.float32)
            us = tf.gather_nd(U, tf.stack([batch_range,zeros ], axis=1))
            ue = tf.gather_nd(U, tf.stack([batch_range, ones], axis=1))

            xi = tf.contrib.layers.xavier_initializer()
            WdS = tf.get_variable(initializer=xi, shape=[5*FLAGS.state_size, FLAGS.state_size],  name="WdS",) # 5l * l

            xi = tf.contrib.layers.xavier_initializer()
            W1S = tf.get_variable(initializer=xi, shape=[3*FLAGS.state_size,FLAGS.state_size, FLAGS.pooling_size],name="W1S")   # 3l X l X p
            b1S = tf.Variable(initial_value=tf.zeros_initializer(shape=(FLAGS.state_size,FLAGS.pooling_size)),name="b1S")

            xi = tf.contrib.layers.xavier_initializer()
            W2S = tf.get_variable(initializer=xi,shape=(FLAGS.state_size,FLAGS.state_size, FLAGS.pooling_size),name="W2S")  # l X l X p
            b2S = tf.Variable(initial_value=tf.zeros_initializer(shape=(FLAGS.state_size,FLAGS.pooling_size)),name="b2S")

            xi = tf.contrib.layers.xavier_initializer()
            W3S = tf.get_variable(initializer=xi,shape=[2* FLAGS.state_size,1, FLAGS.pooling_size],name="W3S")
            b3S = tf.Variable(initial_value=tf.zeros_initializer(shape=(1,FLAGS.pooling_size)),name="b3S")


            xi = tf.contrib.layers.xavier_initializer()
            WdE = tf.get_variable(initializer=xi, shape=[5*FLAGS.state_size, FLAGS.state_size],  name="WdE",) # 5l * l

            xi = tf.contrib.layers.xavier_initializer()
            W1E = tf.get_variable(initializer=xi, shape=[3*FLAGS.state_size,FLAGS.state_size, FLAGS.pooling_size],name="W1E")   # 3l X l X p
            b1E = tf.Variable(initial_value=tf.zeros_initializer(shape=(FLAGS.state_size,FLAGS.pooling_size)),name="b1E")

            xi = tf.contrib.layers.xavier_initializer()
            W2E = tf.get_variable(initializer=xi,shape=(FLAGS.state_size,FLAGS.state_size, FLAGS.pooling_size),name="W2E")  # l X l X p
            b2E = tf.Variable(initial_value=tf.zeros_initializer(shape=(FLAGS.state_size,FLAGS.pooling_size)),name="b2E")

            xi = tf.contrib.layers.xavier_initializer()
            W3E = tf.get_variable(initializer=xi,shape=[2* FLAGS.state_size,1, FLAGS.pooling_size],name="W3E")
            b3E = tf.Variable(initial_value=tf.zeros_initializer(shape=(1,FLAGS.pooling_size)),name="b3E")


            for iter in range(4):
                if iter > 0:
                    tf.get_variable_scope().reuse_variables()

                se = tf.concat(1, [ue, us])

                _, h = cell(se, h, scope="LSTM")

                hse = tf.concat(1, [h[1], us, ue])


                rS = tf.nn.tanh(tf.matmul(hse, WdS))

                RS = tf.reshape(tf.tile(tf.reshape(rS,[-1]),[FLAGS.max_document_size]),[FLAGS.batch_size, -1,FLAGS.state_size])

                URS = tf.concat(2, [U,RS])  #m*batch*3l
                W1S_URS = tf.einsum('ijk,klm->ijlm',URS, W1S)
                M1S = tf.reduce_max(W1S_URS +b1S, axis=3)

                W2S_M1S = tf.einsum('ijk,klm->ijlm',M1S, W2S)
                M2S = tf.reduce_max(W2S_M1S +b2S, axis=3)

                M1S_M2S = tf.concat(2, [M1S,M2S])
                W3S_M1S_M2S = tf.einsum('ijk,klm->ijlm',M1S_M2S, W3S)
                ALPHA = tf.reshape(
                    tf.reduce_max(W3S_M1S_M2S +b3S, axis=3),
                    [FLAGS.batch_size, FLAGS.max_document_size]
                )

                loss += tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(tf.nn.softmax(ALPHA), self.span_placeholder[:,0])
                )

                start_indices = tf.cast(tf.argmax(ALPHA, axis=1), dtype=tf.int32)
                us = tf.gather_nd(U,tf.stack([batch_range, start_indices],1))



                hse = tf.concat(1, [h[1], us, ue])

                rE = tf.nn.tanh(tf.matmul(hse, WdE))

                RE = tf.reshape(tf.tile(tf.reshape(rE,[-1]),[FLAGS.max_document_size]),[FLAGS.batch_size, -1,FLAGS.state_size])

                URE = tf.concat(2, [U,RE])  #m*batch*3l
                W1E_URE = tf.einsum('ijk,klm->ijlm',URE, W1E)
                M1E = tf.reduce_max(W1E_URE +b1E, axis=3)

                W2E_M1E = tf.einsum('ijk,klm->ijlm',M1E, W2E)
                M2E = tf.reduce_max(W2E_M1E +b2E, axis=3)

                M1E_M2E = tf.concat(2, [M1E,M2E])
                W3E_M1E_M2E = tf.einsum('ijk,klm->ijlm',M1E_M2E, W3E)
                BETA = tf.reshape(
                    tf.reduce_max(W3E_M1E_M2E +b3E, axis=3),
                    [FLAGS.batch_size, FLAGS.max_document_size]
                )
                loss += tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(tf.nn.softmax(BETA), self.span_placeholder[:,1])
                )

                end_indices = tf.cast(tf.argmax(BETA, axis=1), dtype=tf.int32)
                ue = tf.gather_nd(U,tf.stack([batch_range, end_indices],1))

        # return (
        #            loss,
        #            tf.shape(U, name="debug_U"),
        #            tf.shape(h, name="debug_h"),
        #            tf.shape(us, name="debug_us"),
        #            tf.shape(ue, name="debug_ue"),
        #            tf.shape(se, name="debug_se"),
        #            tf.shape(hse, name="debug_hse"),
        #            tf.shape(rS, name="debug_rS"),
        #            tf.shape(RS, name="debug_RS"),
        #            tf.shape(URS, name="debug_URS"),
        #            tf.shape(W1S_URS, name="debug_W1S_URS"),
        #            tf.shape(M1S, name="debug_M1S"),
        #            tf.shape(W2S_M1S, name="debug_W2S_M1S"),
        #            tf.shape(M2S, name="debug_M2S"),
        #            tf.shape(M1S_M2S, name="debug_M1S_M2S"),
        #            tf.shape(W3S_M1S_M2S, name="debug_W3_M1S_M2S"),
        #            tf.shape(ALPHA, name="debug_ALPHA"),
        #            tf.shape(start_indices, name="debug_start_indices"),
        #            tf.shape(rE, name="debug_rE"),
        #            tf.shape(RE, name="debug_RE"),
        #            tf.shape(URE, name="debug_URE"),
        #            tf.shape(W1E_URE, name="debug_W1E_URE"),
        #            tf.shape(M1E, name="debug_M1E"),
        #            tf.shape(W2E_M1E, name="debug_W2E_M1E"),
        #            tf.shape(M2E, name="debug_M2E"),
        #            tf.shape(M1E_M2E, name="debug_M1E_M2E"),
        #            tf.shape(W3E_M1E_M2E, name="debug_W3_M1E_M2E"),
        #            tf.shape(BETA, name="debug_beta"),
        #            tf.shape(end_indices, name="debug_end_indices"),
        #         ) + encoded_representation

        return loss/8, tf.stack([start_indices, end_indices], 1)



    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)

        loss = tf.reduce_mean(tf.nn.l2_loss(preds-self.span_placeholder))

        ### END YOUR CODE
        return loss,

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss[0])

        ### END YOUR CODE
        return (train_op,)+ loss

    def predict_on_batch(self, sess, question_batch, document_batch):
        # feed = self.create_feed_dict(
        #     question_batch = question_batch,
        #     document_batch= document_batch
        # )
        # def arr(*args):
        #     return list(args)
        # pred = sess.run( arr(*self.prediction), feed_dict=feed)
        # for i, tensor in enumerate(self.prediction):
        #     if tensor.name.startswith("debug_"):
        #         logger.debug("Shape of {} == {}".format(tensor.name[6:], pred[i]))
        # # print pred[0]
        # return pred
        raise NotImplementedError


    def train_on_batch(self, sess, question_batch, document_batch, span_batch):
        feed = self.create_feed_dict(
            question_batch = question_batch,
            document_batch= document_batch,
            span_batch=span_batch
        )
        def arr(*args):
            return list(args)
        train_op = sess.run( arr(*self.train_op), feed_dict=feed)
        loss = train_op[1]
        pred = train_op[2]
        # for i, tensor in enumerate(self.prediction):
        #     if tensor.name.startswith("debug_"):
        #         logger.debug("Shape of {} == {}".format(tensor.name[6:], pred[i]))
        # print pred[0]
        return loss, pred

    def __init__(self, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.question_placeholder = None
        self.document_placeholder = None
        self.span_placeholder = None
        self.dropout_placeholder = None
        self.build()


def do_test2():
    embeddings = du.load_embeddings()
    # train_questions, train_contexts, train_spans = du.load_dataset(type = "train")
    val_questions, val_contexts, val_spans = du.load_dataset(type = "val")


    with tf.Graph().as_default():

        logger.info("Building model...",)
        start = time.time()
        model = CoattentionModel(embeddings)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
            session.run(init)

            for epoch in range(1):
                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
                logger.info("Epoch %d out of %d", epoch + 1, 1)
                for i in range(int(len(val_questions)/FLAGS.batch_size)):
                    loss, pred = model.train_on_batch(
                        session,
                        val_questions[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size],
                        val_contexts[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size],
                        val_spans[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size],
                    )
                    print i, loss, pred, val_spans[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size]
            train_writer.close()

    logger.info("Model did not crash!")
    logger.info("Passed!")

if __name__ == "__main__":
    parse_args.parse_args()
    do_test2()