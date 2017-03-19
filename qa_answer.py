from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
from os.path import join as pjoin
import logging
from util import Progbar

from tqdm import tqdm
import tensorflow as tf

from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url,tokenize
import qa_data
from qa_data_util import *
from  parse_args import parse_args
import evaluate

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


# def initialize_vocab(vocab_path):
#     if tf.gfile.Exists(vocab_path):
#         rev_vocab = []
#         with tf.gfile.GFile(vocab_path, mode="rb") as f:
#             rev_vocab.extend(f.readlines())
#         rev_vocab = [line.strip('\n') for line in rev_vocab]
#         vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
#         return vocab, rev_vocab
#     else:
#         raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens, _, _ = tokenize(context, tokenizer=FLAGS.tokenizer)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens, _, _ = tokenize(question, tokenizer=FLAGS.tokenizer)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(session,model, dataset, rev_vocab):
    answers = {}
    num_dev_batches = int(len(dataset['q'])/FLAGS.batch_size) + 1
    prog = Progbar(target=num_dev_batches)
    for i in range(num_dev_batches):
        data_batch = get_batch(dataset, i)
        pred = model.predict_on_batch(sess=session, data_batch=data_batch)
        for j,document in enumerate(data_batch['c']):
            answers[data_batch['q_uuids'][j]] = " ".join([rev_vocab[document[index]] for index in pred[j]])

        prog.update(i+1, [])

    return answers


def main(_):
    vocab,rev_vocab = initialize_vocab()
    embeddings = load_embeddings()

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    contexts, questions, question_uuids = prepare_dev(dev_dirname, dev_filename, vocab)

    questions = [qa_data.basic_tokenizer(records) for records in questions]

    contexts = [qa_data.basic_tokenizer(records) for records in contexts]

    questions, questions_mask, questions_seq = padding(cast_to_int(questions), FLAGS.max_question_size)
    contexts, contexts_mask, contexts_seq = padding(cast_to_int(contexts), FLAGS.max_document_size)

    dataset = {
        'q': questions,
        'q_m': questions_mask,
        'q_s': questions_seq,
        'c': contexts,
        'c_m': contexts_mask,
        'c_s': contexts_seq,
        'q_uuids':question_uuids
    }
    print("lenght of dev set: {}".format(len(dataset['q'])))
    # ========= Model-specific =========
    # You must change the following code to adjust to your model
    model = choose_model(embeddings=embeddings)

    with tf.Session() as sess:
        restore_model(session=sess, run_id=FLAGS.run_id)
        answers = generate_answers(sess, model, dataset, rev_vocab=rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))

     print(json.dumps(evaluate(json.load(dev_dirname), answers)))



if __name__ == "__main__":
    parse_args()
    tf.app.run()
