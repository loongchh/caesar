import logging
import tensorflow as tf
import numpy as np
from os.path import join as pjoin
from datetime import datetime
import os

import parse_args
FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def choose_model(embeddings, debug=False):
    model = None
    if FLAGS.model.lower() == "match_lstm":
        from match_lstm import MatchLstmModel
        model = MatchLstmModel(embeddings, debug)
    elif FLAGS.model.lower() == "match_lstm_boundry":
        from match_lstm_boundry import MatchLstmBoundryModel
        model = MatchLstmBoundryModel(embeddings, debug)
    elif FLAGS.model.lower() == "coattention":
        from coattention import CoattentionModel
        model = CoattentionModel(embeddings, debug)

    return model


def checkpoint_model(session,run_id, version=1):
    saver = tf.train.Saver()
    save_dir = pjoin(FLAGS.train_dir, FLAGS.model, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = saver.save(session,pjoin(save_dir, "model-{}.ckpt".format(version)))
    logger.info("Model saved at: %s" % save_path)


def restore_model(session, run_id, version=1):
    saver = tf.train.Saver()
    save_path = pjoin(FLAGS.train_dir, FLAGS.model, run_id, "model-{}.ckpt".format(version))
    saver.restore(session, save_path)
    logger.info("Model restored from: {}".format(save_path))


def load_embeddings():
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)['glove']
    embeddings=embeddings.astype(np.float32)

    zeros = np.sum([1 for x in embeddings if np.all(x==0)])
    logger.info("Loaded GloVe embeddings of {} vocabs with {} zero vectors.".format(len(embeddings), zeros))

    return embeddings


def read_dataset(filename, truncate_length=10000):
    with open(filename, 'r') as f:
        data = f.readlines()
    return [x.strip().split(" ")[:truncate_length] for x in data]


def load_dataset(type='train', plot=False, debug=False):
    data_dir = FLAGS.data_dir
    train_path_q = pjoin(data_dir, "{}.ids.question".format(type))
    train_path_c = pjoin(data_dir, "{}.ids.context".format(type))
    train_path_a = pjoin(data_dir, "{}.span".format(type))
    questions = read_dataset(train_path_q)
    contexts = read_dataset(train_path_c)
    spans = read_dataset(train_path_a)

    # Assert data length
    assert len(questions) == len(contexts) and len(contexts) == len(spans)
    logger.info("Loaded {} dataset of size {}.".format(type, len(questions)))

    # cast the data from string to int
    questions = cast_to_int(questions)
    contexts = cast_to_int(contexts)
    spans = cast_to_int(spans)

    # Flatten Answer span to obtain Ground Truth
    ground_truth = get_answer_from_span(spans)
    if debug:
        logger.debug("Sample Span: {}".format(spans[0]))
        logger.debug("Flattened Answer from span: {}".format(ground_truth[0]))

    if plot:
        plot_histogram(questions, "{}-questions".format(type))
        plot_histogram(contexts, "{}-contexts".format(type))
        plot_histogram(ground_truth, "{}-answers".format(type))

    questions, contexts,spans, ground_truth = filter_data(questions, contexts, spans, ground_truth)

    if debug:
        logger.debug("Filtered {} data, new size {}.".format(type, len(questions)))
    if plot:
        plot_histogram(contexts, "{}-contexts-filtered".format(type))
        plot_histogram(questions, "{}-questions-filtered".format(type))
        plot_histogram(ground_truth, "{}-answers-filtered".format(type))

    questions, questions_mask, questions_seq = padding(questions, FLAGS.max_question_size)
    contexts, contexts_mask, contexts_seq = padding(contexts, FLAGS.max_document_size)
    answers, answers_mask, answers_seq = padding(ground_truth,FLAGS.max_answer_size, zero_vector=FLAGS.max_document_size)

    if plot:
        plot_histogram(contexts, "{}-contexts-padded".format(type))
        plot_histogram(questions, "{}-questions-padded".format(type))
        plot_histogram(answers, "{}-answers-padded".format(type))

    data = {
        'q': questions,
        'q_m': questions_mask,
        'q_s': questions_seq,
        'c': contexts,
        'c_m': contexts_mask,
        'c_s': contexts_seq,
        's': spans,
        'gt': ground_truth,
        's_e': answers,
        'a': answers,
        'a_m': answers_mask,
        'a_s': answers_seq,
    }
    return data


def cast_to_int(data):
    return [[int(field) for field in record] for record in data]


def filter_data(questions, contexts, spans, exploded_spans):

    def filter(q_len, c_len, a_len=1):
        filter1 = FLAGS.min_question_size < q_len <= FLAGS.max_question_size
        filter2 = FLAGS.min_document_size < c_len <= FLAGS.max_document_size
        filter3 = FLAGS.min_answer_size < a_len <= FLAGS.max_answer_size
        return filter1 and filter2 and filter3

    indices = [i for i, q in enumerate(questions) if filter(len(q), len(contexts[i]), len(exploded_spans[i])) ]

    return (
        [questions[i] for i in indices],
        [contexts[i] for i in indices],
        [spans[i] for i in indices],
        [exploded_spans[i] for i in indices]
    )


def get_answer_from_span(spans):

    def fun(s, e):
        s,e = (s, e) if s <= e else (e, s)
        return range(s,e+1)
    return [fun(s[0], s[1]) for s in spans]


def padding(data, max_length, zero_vector=0):

    # clip records to max length
    data = [record[:max_length] for record in data]

    # sequence length vector
    seq = [len(record) for record in data]

    # Masking vectors
    mask = [len(record)*[True] + (max_length - len(record))*[False] for record in data]

    # padded data
    data = [record[:] + (max_length - len(record))*[zero_vector] for record in data]

    return data, mask,seq


def plot_histogram(data,name ):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        data_lengths = [len(x) for x in data]
        logger.debug("max length for {} = {}".format(name,max(data_lengths)))
        plt.clf()
        plt.hist(data_lengths,bins=50)
        plt.title("Histogram: {}".format(name))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        output_path = pjoin("../plots/","{}-histogram.png".format(name))
        plt.savefig(output_path)
    except ImportError:
        pass

def initialize_vocab():
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_batch(data, i, permutation=None):
    start = i*FLAGS.batch_size
    end = (i+1)*FLAGS.batch_size

    if permutation is not None:
        indices = permutation[start:end]
    else:
        indices = range(start, end)

    batch = {}
    for k in data:
        batch[k] = [data[k][idx] for idx in indices]

    return batch


def test_get_batch():
    data = {
        "q": [[1, 2, 3]]*FLAGS.batch_size + [[3, 4, 6]]*FLAGS.batch_size
    }
    # test without permutation
    assert get_batch(data,1) == {"q": [[3,4,6]]*FLAGS.batch_size}

    # test with simple permutation
    permutation = range(FLAGS.batch_size,2*FLAGS.batch_size)+range(0,FLAGS.batch_size)
    actual = get_batch(data, 1, permutation=permutation)
    expected = {"q": [[1,2,3]]*FLAGS.batch_size}
    assert actual == expected

    # test with random permutation
    permutation =np.random.permutation(2*FLAGS.batch_size)
    actual = get_batch(data, 1, permutation=permutation)
    expected = {"q": [[1, 2, 3] if idx < FLAGS.batch_size else [3, 4, 6] for i, idx in enumerate(permutation) if i >= FLAGS.batch_size]}
    assert actual == expected

if __name__ == '__main__':
    parse_args.parse_args()
    test_get_batch()
    exit()
    embeddings = load_embeddings()
    vocab, rev_vocab = initialize_vocab()
    # for word in vocab:
    #     if word[0].islower():
    #         w = word[0].upper() + word[1:]
    #         if w in vocab:
    #             embeddings[vocab[w]] = embeddings[vocab[word]]
    #
    #
    #
    # print embeddings[vocab['Who']]
    # exit()
    train_data = load_dataset(type = "train", plot=True)
    val_data = load_dataset(type = "val", plot=True)
