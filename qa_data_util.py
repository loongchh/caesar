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

    # Truncate context paragraph to be of size FLAGS.max_document_size,
    # and questions to FLAGS.max_question_size
    contexts, contexts_mask = clip_and_pad(contexts, FLAGS.max_document_size)
    questions, questions_mask = clip_and_pad(questions, FLAGS.max_question_size)

    if plot:
        plot_histogram(contexts, "{}-contexts-truncated".format(type))
        plot_histogram(questions, "{}-questions-truncated".format(type))

    return questions, contexts, spans


def clip_and_pad(data, max_length, zero_vector = 0):
    mask = [min(len(record), max_length)*[True] + (max_length - len(record))*[False] for record in data]
    data = [record[:max_length] + (max_length - len(record))*[zero_vector] for record in data]
    return data, mask

def test_clip_and_pad():
    print clip_and_pad([[0,1,2,3], [4,5,6],[1],[]], 2)


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
    output_path = pjoin("plots/","{}-histogram.png".format(name))
    plt.savefig(output_path)

if __name__ == '__main__':
    parse_args.parse_args()
    test_clip_and_pad()
    embeddings = load_embeddings()
    train_questions, train_contexts, train_spans = load_dataset(type = "train", plot=True)
    val_questions, val_contexts, val_spans = load_dataset(type = "val", plot=True)