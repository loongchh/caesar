from tensorflow.python.platform import gfile
import gzip
from os import listdir
from os.path import isfile, join
import time
import logging
import numpy as np
import os

logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def java_string_hashcode(s):
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFF
    return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def read_partitioned_glove(new_words):
    missing_vocab_embedding = {}
    for partition in range(1000):
        path = 'glove_840_partitioned/h=%s' % partition
        onlyfiles = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
        print(onlyfiles[0])
        with gzip.open(onlyfiles[0], 'r') as fh:
                for line in fh.readlines():
                    array = line.lstrip().rstrip().split(" ")
                    if array[0] in new_words[partition]:
                        missing_vocab_embedding[array[0]] = array[1:]
                        new_words[partition].remove(array[0])
                    if len(new_words[partition]) <=0:
                        break
    return missing_vocab_embedding


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def partition_missing_words(missing_words):
    partitioned_missing_words = [[] for i in range(1000)]
    for word in missing_words:
        partition = abs(java_string_hashcode(word))%1000
        partitioned_missing_words[partition].append(word)
    return partitioned_missing_words


def enhance_vocabulary(vocab, rev_vocab, embedding, missing_words, glove_dim=300):
    partitioned_missing_words = partition_missing_words(missing_words)
    embeddings_found, elapsed = timed(lambda: read_partitioned_glove(partitioned_missing_words))

    random_init = np.random.randn(len(missing_words)-len(embeddings_found),glove_dim)

    logger.info('Took %d secs for embedding lookup' % elapsed)
    original_size = len(vocab)
    delta_embedding = []
    for i, word in enumerate(missing_words):
        vocab[word] = str(i + original_size)
        rev_vocab.append(word)
        if word in embeddings_found:
            delta_embedding.append(np.array(embeddings_found[word], dtype=np.float32))
        else:
            a = np.random.random_sample(glove_dim)
            delta_embedding.append(a)

    return vocab, rev_vocab, np.append(embedding, delta_embedding,axis=0), len(embeddings_found)




def timed(f):
  start = time.time()
  ret = f()
  elapsed = time.time() - start
  return ret, elapsed


def load_embeddings_for_test():
    embed_path1 = os.path.join("data_6B_100", "squad", "glove.trimmed.840B.300.npz")
    embeddings1 = np.load(embed_path1)['glove']
    embeddings1=embeddings1.astype(np.float32)

    embed_path2 = os.path.join("data_6B_100_dev", "squad", "glove.trimmed.840B.300.npz")
    embeddings2 = np.load(embed_path2)['glove']
    embeddings2=embeddings2.astype(np.float32)

    return embeddings1, embeddings2

def test_enhance_vocabulary():
    vocab1, rev_vocab1 = initialize_vocabulary("data_840B_300/squad/vocab.dat")
    vocab2, rev_vocab2 = initialize_vocabulary("data_840B_300_dev/squad/vocab.dat")

    embeddings1, embeddings2 = load_embeddings_for_test()

    original_matches = sum([ 1 if np.all(embeddings1[i] == embeddings2[vocab2[word]]) else 0 for i,word in enumerate(rev_vocab1)])

    missing_words = []
    for word in vocab2:
        if word not in vocab1:
            missing_words.append(word)

    vocab1_new,rev_vocab1_new, embeddings1_new,found = enhance_vocabulary(vocab1, rev_vocab1, embeddings1, missing_words, 100)

    assert len(embeddings1_new) == len(embeddings2)
    matches = sum([ 1 if np.all(embeddings1_new[i] == embeddings2[vocab2[word]]) else 0 for i,word in enumerate(rev_vocab1_new)])
    print original_matches, matches, found
    assert original_matches+found == matches

if __name__ == '__main__':
    test_enhance_vocabulary()