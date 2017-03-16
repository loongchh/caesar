from __future__ import print_function
import argparse
import json
import linecache
import nltk
import numpy as np
import os
import sys
from tqdm import tqdm
import random
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

from collections import Counter
from six.moves.urllib.request import urlretrieve

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(42)
np.random.seed(42)

squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

# Size train: 30288272
# size dev: 4854279

def reporthook(t):
  """https://github.com/tqdm/tqdm"""
  last_b = [0]

  def inner(b=1, bsize=1, tsize=None):
    """
    b: int, optional
        Number of blocks just transferred [default: 1].
    bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
    tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
    """
    if tsize is not None:
        t.total = tsize
    t.update((b - last_b[0]) * bsize)
    last_b[0] = b
  return inner

def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix,filename), reporthook=reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix,filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def data_from_json(filename):
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def list_topics(data):
    list_topics = [data['data'][idx]['title'] for idx in range(0,len(data['data']))]
    return list_topics


def tokenize(sequence, tokenizer="CORE-NLP"):
    if tokenizer == "CORE-NLP":
        output = nlp.annotate(sequence.encode('utf-8'), properties={'annotators': 'tokenize,ssplit','outputFormat': 'json'})
        if isinstance(output, unicode):
            output = json.loads(output[:1543]+output[1544:1652]+output[1654:])
        tokens = []
        char_idx_to_token_idx_map = {}
        token_count = 0;
        for i in range(len(output['sentences'])):
            for t in output['sentences'][i]['tokens']:
                char_idx_to_token_idx_map[t['characterOffsetBegin']] = token_count;
                tokens.append(t['word'].encode('utf-8'))
                token_count += 1
        return tokens, char_idx_to_token_idx_map
    else:
        tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
        return map(lambda x:x.encode('utf8'), tokens), {}


def token_idx_map(context, context_tokens):
    print(context)
    print(context_tokens)
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):
        if char != u' ':
            acc += char
            context_token = unicode(context_tokens[current_token_idx])
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    print(token_map)
    exit()
    return token_map


def invert_map(answer_map):
    return {v[1]: [v[0], k] for k, v in answer_map.iteritems()}


def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(os.path.join(prefix, tier +'.context'), 'w') as context_file,  \
         open(os.path.join(prefix, tier +'.question'), 'w') as question_file,\
         open(os.path.join(prefix, tier +'.answer'), 'w') as text_file, \
         open(os.path.join(prefix, tier +'.span'), 'w') as span_file:

        print(len(dataset['data']))

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens, context_char_idx_to_token_idx_map = tokenize(context)
                # answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens, _ = tokenize(question)

                    answers = qas[qid]['answers']
                    qn += 1

                    num_answers = range(1)

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']
                        a_s = qas[qid]['answers'][ans_id]['answer_start']

                        text_tokens, _ = tokenize(text)

                        answer_start_char_idx = qas[qid]['answers'][ans_id]['answer_start']
                        span_start = context_char_idx_to_token_idx_map[answer_start_char_idx]
                        span_end = span_start + len(text_tokens) -1

                        # print(context_tokens[span_start])
                        # print(context_tokens[span_end])
                        #
                        # print(text_tokens)
                        #
                        # exit()

                        # remove length restraint since we deal with it later
                        context_file.write(' '.join(context_tokens) + '\n')
                        question_file.write(' '.join(question_tokens) + '\n')
                        text_file.write(' '.join(text_tokens) + '\n')
                        span_file.write(' '.join([str(span_start), str(span_end)]) + '\n')

                        # except Exception as e:
                        #     skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn,an


def save_files(prefix, tier, indices):
  with open(os.path.join(prefix, tier + '.context'), 'w') as context_file,  \
     open(os.path.join(prefix, tier + '.question'), 'w') as question_file,\
     open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
     open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

    for i in indices:
      context_file.write(linecache.getline(os.path.join(prefix, 'train.context'), i))
      question_file.write(linecache.getline(os.path.join(prefix, 'train.question'), i))
      text_file.write(linecache.getline(os.path.join(prefix, 'train.answer'), i))
      span_file.write(linecache.getline(os.path.join(prefix, 'train.span'), i))


def split_tier(prefix, train_percentage = 0.9, shuffle=False):
    # Get number of lines in file
    context_filename = os.path.join(prefix, 'train' + '.context')
    # Get the number of lines
    with open(context_filename) as current_file:
        num_lines = sum(1 for line in current_file)
    # Get indices and split into two files
    indices_dev = range(num_lines)[int(num_lines * train_percentage)::]
    if shuffle:
        np.random.shuffle(indices_dev)
        print("Shuffling...")
    save_files(prefix, 'val', indices_dev)
    indices_train = range(num_lines)[:int(num_lines * train_percentage)]
    if shuffle:
        np.random.shuffle(indices_train)
    save_files(prefix, 'train', indices_train)


if __name__ == '__main__':

    download_prefix = os.path.join("download", "squad")
    data_prefix = os.path.join("data", "squad")

    print("Downloading datasets into {}".format(download_prefix))
    print("Preprocessing datasets into {}".format(data_prefix))

    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"

    maybe_download(squad_base_url, train_filename, download_prefix, 30288272L)

    train_data = data_from_json(os.path.join(download_prefix, train_filename))

    train_num_questions, train_num_answers = read_write_dataset(train_data, 'train', data_prefix)

    # In train we have 87k+ questions, and one answer per question.
    # The answer start range is also indicated

    # 1. Split train into train and validation into 95-5
    # 2. Shuffle train, validation
    print("Splitting the dataset into train and validation")
    split_tier(data_prefix, 0.95, shuffle=True)

    print("Processed {} questions and {} answers in train".format(train_num_questions, train_num_answers))

    print("Downloading {}".format(dev_filename))
    dev_dataset = maybe_download(squad_base_url, dev_filename, download_prefix, 4854279L)

    # In dev, we have 10k+ questions, and around 3 answers per question (totaling
    # around 34k+ answers).
    # dev_data = data_from_json(os.path.join(download_prefix, dev_filename))
    # list_topics(dev_data)
    # dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', data_prefix)
    # print("Processed {} questions and {} answers in dev".format(dev_num_questions, dev_num_answers))
