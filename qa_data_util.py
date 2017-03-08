import logging
import tensorflow as tf
import numpy as np
from os.path import join as pjoin

import parse_args
FLAGS = tf.app.flags.FLAGS


logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_embeddings():
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.vocab_dim))
    embeddings = np.load(embed_path)['glove']
    logger.debug("loaded glove embeddings of vocab size: {}".format(len(embeddings)))
    return embeddings


def read_dataset(filename, truncate_length=10000):
    with open(filename, 'r') as f:
        data = f.readlines()
    return [x.strip().split(" ")[:truncate_length] for x in data]


def load_dataset(type='train', plot=False):
    data_dir = FLAGS.data_dir
    train_path_q = pjoin(data_dir, "{}.ids.question".format(type))
    train_path_c = pjoin(data_dir, "{}.ids.context".format(type))
    train_path_a = pjoin(data_dir, "{}.span".format(type))
    questions = read_dataset(train_path_q)
    contexts = read_dataset(train_path_c)
    spans = read_dataset(train_path_a)
    assert len(questions) == len(contexts) and  len(contexts) == len(spans)
    logger.debug("loaded {} data of size {}".format(type, len(questions)))

    if plot:
        plot_histogram(questions, "{}-questions".format(type))
        plot_histogram(contexts, "{}-contexts".format(type))

    questions, contexts,spans = filter_data(questions, contexts, spans)

    logger.debug("filtered {} data, new size {}".format(type, len(questions)))
    if plot:
        plot_histogram(contexts, "{}-contexts-filtered".format(type))
        plot_histogram(questions, "{}-questions-filtered".format(type))


    questions, questions_mask, questions_seq = padding(questions, 15)
    contexts, contexts_mask, contexts_seq = padding(contexts, 120)

    if plot:
        plot_histogram(contexts, "{}-contexts-truncated".format(type))
        plot_histogram(questions, "{}-questions-truncated".format(type))
    # print "hello", spans[0]
    spans = sparse_span_matrix(spans)
    # print "hello", spans[0]
    data = {
        'q': questions,
        'q_m': questions_mask,
        'q_s': questions_seq,
        'c': contexts,
        'c_m': contexts_mask,
        'c_s': contexts_seq,
        's': spans}
    return data


def filter_data(questions, contexts, spans):

    def filter(q_len, c_len):
        return 5 < q_len <= 15 and 80 < c_len <= 120

    indices = [i for i, q in enumerate(questions) if filter(len(q), len(contexts[i])) ]

    return (
        [questions[i] for i in indices],
        [contexts[i] for i in indices],
        [spans[i] for i in indices]
    )

def sparse_span_matrix(span):
    return [[int(s[0]), (int(s[1])-int(s[0]))* FLAGS.max_document_size + int(s[1])]for s in span]


def padding(data, max_length, zero_vector=0):
    seq = [len(record) for record in data]
    mask = [min(len(record), max_length)*[True] + (max_length - len(record))*[False] for record in data]
    data = [record[:max_length] + (max_length - len(record))*[zero_vector] for record in data]

    return data, mask,seq


def plot_histogram(data,name ):
    # import only if required
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


def get_batch(data, i):
    start = i*FLAGS.batch_size
    end = (i+1)*FLAGS.batch_size

    batch = {}
    for k in data:
        batch[k] = data[k][start:end]

    return batch

if __name__ == '__main__':
    parse_args.parse_args()
    # test_clip_and_pad()
    embeddings = load_embeddings()
    train_data = load_dataset(type = "train", plot=True)
    val_data = load_dataset(type = "val", plot=True)