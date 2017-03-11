"""Sequence-to-sequence model with an attention mechanism."""
# see https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
# compare https://github.com/tflearn/tflearn/blob/master/examples/nlp/seq2seq_example.py
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
# from https://gist.github.com/pannous/b3f8ab944a85b33e694de21c6ded029e
# see https://www.tensorflow.org/versions/master/tutorials/seq2seq/

####################################################################
### CONFIG SECTION ###
####################################################################

# the vocab size is an input parameter to at least one of tensorflow's functions,
# in fact, it needs the vocab size for the encoder and for the decoder
# (so each may have a specific vocab size).
# for this example below, the vocabulary are the 256 ASCII characters:
# standard ASCII characters(0-127) and Extended ASCII characters(128-255).
# Basically, we have one *class* per character/byte.
vocab_size = 256
target_vocab_size = vocab_size

learning_rate = 0.1

# our input and response words can be up to 10 characters long.
# setting it to one bucket is a simple way to use the bucketing, but not really getting value for it
# if just using one bucket.  In this case, just setting it to the max for each, input and output
# sequences.
buckets = [(10, 10)]
# fill words shorter than 10 characters with 'padding' zeroes
PAD = [0]
batch_size = 10

# input sequence of characters, padded to maximum sequence length
# btw, I dont see reversing of the input here, as recommended in other cases, mostly because
# there is no lexical relationship between this input and output sequence (unlike in a
# question-answer pair)
# here's what it looks like in bytes:
# <type 'list'>: [104, 101, 108, 108, 111, 0, 0, 0, 0, 0]
input_data = [map(ord, "hello") + PAD * 5] * batch_size

# output sequence of characters, padded to max sequence length
# Some examples online discuss appending a prefix symbol the the decoder inputs (e.g. GO),
# but I dont see that done here, and also missing why that is really needed.
# here's what it looks like in bytes:
# <type 'list'>: [119, 111, 114, 108, 100, 0, 0, 0, 0, 0]
# the 0 for the null character actually makes it learn a lot faster, causing it
# to output zero for the rest of the sequence (need to review those latching/toggling
# rnn exercises from the homework)
target_data = [map(ord, "world\0") + PAD * 4] * batch_size

# mask padding.
# here we want the last byte predicted to be zero, which is why have 6 and not 5 '1.0's below,
# so we are counting the '\0' null character as an element of the target data that we
# want to learn.
target_weights = [[1.0] * 6 + [0.0] * 4] * batch_size

####################################################################
### MODEL SECTION ###
####################################################################

class BabySeq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, batch_size):
        # need some clean up on member variables, some times they are used in the code below,
        # other times the closures (input arguments to __init__) are used.
        self.buckets = buckets
        self.batch_size = batch_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

        cell = single_cell = tf.nn.rnn_cell.GRUCell(size)
        # this is for stacking lstm's, if choosing to do so
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        # feeds for target and masks
        for i in xrange(buckets[-1][1]):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        targets = [self.decoder_inputs[i] for i in xrange(len(self.decoder_inputs))]

        #TODO: why does it need both the decoder_inputs and targets as inputs below, if they are identical?
        #TODO: is this something needed for attention?
        #TODO: do they need to be shifted as in hello_sequece_orig.py?
        self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False))

        # Gradient update operations for training the model.
        params = tf.trainable_variables()
        self.updates = []
        for b in xrange(len(buckets)):
            self.updates.append(tf.train.AdamOptimizer(learning_rate).minimize(self.losses[b]))

        # save once all the variables have been defined.
        self.saver = tf.train.Saver(tf.all_variables())

    # performs a forward propagation step
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, test):
        # in this case, can just select 0, which is the only bucket, but in a more general
        # sense, would programatically select the bucket for the appropriate lengths of the
        # passed in input and output sequences.
        bucket_id = 0
        encoder_size, decoder_size = self.buckets[bucket_id]

        # Input feed: encoder inputs, decoder inputs, target_weights (masks), as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Output feed: depends on whether we do a backward step or not.
        if not test:
            output_feed = [self.updates[bucket_id], self.losses[bucket_id]]
        else:
            # Loss for this batch.
            output_feed = [self.losses[bucket_id]]

            # Output logits.
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)

        if not test:
            # Gradient norm, loss
            return outputs[0], outputs[1]
        else:
            # loss, outputs.
            return outputs[0], outputs[1:]


# convert byte list to string
def decode(bytes):
    # here why need to do this replace of '\x00' with ''? is that for the go and new line symbols?
    return "".join(map(chr, bytes)).replace('\0', '')


def test():
    # perplexity is related to the cross entropy loss, what was that relationship again?
    perplexity, outputs = model.step(session, input_data, target_data, target_weights, test=True)
    words = np.argmax(outputs, axis=2)  # shape (10, 10, 256)
    word = decode(words[0])
    print("step %d, perplexity %f, output: hello %s?" % (step, perplexity, word))
    if word == "world":
        print(">>>>> success! hello " + word + "! <<<<<<<")
        exit()


step = 0
test_step = 1
with tf.Session() as session:
    session = tf_debug.LocalCLIDebugWrapperSession(session)
    session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    model = BabySeq2Seq(vocab_size, target_vocab_size, buckets, size=10, num_layers=1, batch_size=batch_size)
    session.run(tf.initialize_all_variables())
    while True:
        model.step(session, input_data, target_data, target_weights, test=False)  # no outputs in training
        if step % test_step == 0:
            test()
        step = step + 1
