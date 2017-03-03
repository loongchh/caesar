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
        self.dropout_placeholder = tf.placeholder(tf.float64,
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
        print len(self.pretrained_embeddings)
        print len(self.pretrained_embeddings[0])
        all_embeddings = tf.get_variable("all_embeddings", initializer=self.pretrained_embeddings, dtype=tf.float64)
        question_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.question_placeholder)
        document_embeddings = tf.nn.embedding_lookup(params=all_embeddings, ids=self.document_placeholder)


        ### END YOUR CODE
        return question_embeddings, document_embeddings

    def add_prediction_op(self):
        q,d = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        with tf.variable_scope("BLSTM"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=FLAGS.state_size, forget_bias=1.0)

            initial_state_fw = cell_fw.zero_state(FLAGS.batch_size, tf.float64)
            initial_state_bw = cell_bw.zero_state(FLAGS.batch_size, tf.float64)

            ## calculate document representation

            d = tf.transpose(d, [1, 0, 2])
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
            q = tf.transpose(q, [1, 0, 2])
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
            dtype=tf.float64
        )
        bq = tf.Variable(
            tf.zeros_initializer(
                shape=( 2 * FLAGS.state_size,),
                dtype=tf.float64
            ),
            name="bq"
        )
        # q_representation = tf.nn.tanh(tf.einsum('ijk,kk->ijk',q_representation, Wq) + bq)


        assert d_representation.get_shape().as_list() == [FLAGS.batch_size, FLAGS.max_document_size, 2 * FLAGS.state_size], \
            "document representation are not of the right shape. Expected {}, got {}".format([FLAGS.batch_size, FLAGS.max_document_size, 2 * FLAGS.state_size], d_representation.get_shape().as_list())

        assert q_representation.get_shape().as_list() == [FLAGS.batch_size, FLAGS.max_question_size, 2 * FLAGS.state_size], \
            "questions representation are not of the right shape. Expected {}, got {}".format([FLAGS.batch_size, FLAGS.max_question_size, 2 * FLAGS.state_size], q_representation.get_shape().as_list())

        a = q_representation
        b = tf.transpose(d_representation,perm=[0,2,1])
        # return tf.shape(a),tf.shape(b)
        # exit()

        ###  Weirdo Tensorflow
        # L = tf.batch_matmul( tf.transpose(d_representation,perm=[0,2,1]), q_representation)

        L = tf.batch_matmul(a,b)

        # L = tf.einsum ('ijk,ikl->ijl', q_representation, tf.transpose(d_representation,perm=[0,2,1]))   # batch_size * max_document_size * max_question_size
        Aq = tf.nn.softmax(L) # batch_size * max_document_size * max_question_size
        Ad = tf.nn.softmax(tf.transpose(L, perm=[0,2,1]))   # batch_size *  max_question_size * max_document_size *
        Cq = tf.matmul(d_representation, Aq)  # batch_size * 300 * 60
        Cd = tf.matmul(q_representation, Ad)
        #
        #

        return Aq,Ad

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

        loss = tf.Variable(100.0)

        ### END YOUR CODE
        return loss

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

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        ### END YOUR CODE
        return train_op

    def predict_on_batch(self, sess, question_batch, document_batch):
        feed = self.create_feed_dict(
            question_batch = question_batch,
            document_batch= document_batch
        )
        d_representation, q_representation = sess.run([self.d_representation, self.q_representation], feed_dict=feed)
        return d_representation, q_representation


    def train_on_batch(self, sess, question_batch, document_batch, span_batch):
        feed = self.create_feed_dict(
            question_batch = question_batch,
            document_batch= document_batch,
            span_batch= span_batch
        )
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

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
            session.run(init)
            for epoch in range(1):
                logger.info("Epoch %d out of %d", epoch + 1, 1)
                a,b = model.predict_on_batch(
                    session,
                    val_questions[:FLAGS.batch_size],
                    val_contexts[:FLAGS.batch_size]
                )

                print a, b

    logger.info("Model did not crash!")
    logger.info("Passed!")

if __name__ == "__main__":
    parse_args.parse_args()
    do_test2()