
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
import qa_data_util as du
import another_data_util as adu
from  parse_args import parse_args
import evaluate
logging.getLogger("requests").setLevel(logging.WARNING)
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


def find_missing_words(vocab2,vocab):
    missing_word = []
    for word in vocab2:
        if word not in vocab:
            missing_word.append(word)

    return missing_word



def get_raw_tokens(dataset, tier, vocab, rev_vocab, embeddings):
    vocab2 = {}
    context_maps = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):



            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens, _, _ = tokenize(context, tokenizer=FLAGS.tokenizer)

            for token in context_tokens:
                vocab2[token] = 1
            context_map = {'context_tokens': context_tokens, 'question_maps':[]}

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens, _, _ = tokenize(question, tokenizer=FLAGS.tokenizer)
                question_uuid = qas[qid]['id']

                for token in question_tokens:
                    vocab2[token] = 1
                question_map = {'question_tokens': question_tokens, 'question_uuid':question_uuid }
                context_map['question_maps'].append(question_map)

            context_maps.append(context_map)

    if FLAGS.word_lookup:
        missing_words = find_missing_words(vocab2, vocab)
        vocab, rev_vocab, embeddings, _ = adu.enhance_vocabulary(vocab, rev_vocab, embeddings, missing_words)

    return context_maps, vocab, rev_vocab, embeddings


def read_dataset(dataset, tier, vocab, rev_vocab, embeddings):
    contexts = []
    questions = []
    question_uuids = []

    context_maps, vocab, rev_vocab, embeddings = get_raw_tokens(dataset, tier, vocab, rev_vocab, embeddings)

    for context_map in context_maps:
        question_maps = context_map['question_maps']
        for question_map in question_maps:
            contexts.append(vocab.get(token, qa_data.UNK_ID) for token in context_map['context_tokens'])
            questions.append(vocab.get(token, qa_data.UNK_ID) for token in question_map['question_tokens'])
            question_uuids.append(question_map['question_uuid'])

    return contexts, questions, question_uuids, vocab, rev_vocab, embeddings

def prepare_dev(prefix, dev_filename, vocab, rev_vocab, embeddings):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    contexts, questions, question_uuids, vocab, rev_vocab, embeddings = read_dataset(dev_data, 'dev', vocab, rev_vocab, embeddings)

    return contexts, questions, question_uuids, vocab, rev_vocab, embeddings


def generate_answers(session,model, dataset, rev_vocab):
    answers = {}
    num_dev_batches = int(len(dataset['q'])/FLAGS.batch_size) + 1
    prog = Progbar(target=num_dev_batches)
    for i in range(num_dev_batches):
        data_batch = du.get_batch(dataset, i)
        pred = model.predict_on_batch(sess=session, data_batch=data_batch, rev_vocab=rev_vocab)
        for j,document in enumerate(data_batch['c']):
            answers[data_batch['q_uuids'][j]] = " ".join([rev_vocab[document[index]] for index in pred[j]])

        prog.update(i+1, [])

    return answers


def main(_):
    vocab, rev_vocab = du.initialize_vocab()
    embeddings = du.load_embeddings()

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
    contexts, questions, question_uuids, vocab, rev_vocab, embeddings = prepare_dev(dev_dirname, dev_filename, vocab, rev_vocab, embeddings)

    questions, questions_mask, questions_seq = du.padding(du.cast_to_int(questions), FLAGS.max_question_size)
    contexts, contexts_mask, contexts_seq = du.padding(du.cast_to_int(contexts), FLAGS.max_document_size)

    dataset = {
        'q': questions,
        'q_m': questions_mask,
        'q_s': questions_seq,
        'c': contexts,
        'c_m': contexts_mask,
        'c_s': contexts_seq,
        'q_uuids':question_uuids
    }
    print("length of dev set: {}".format(len(dataset['q'])))
    # ========= Model-specific =========
    # You must change the following code to adjust to your model
    print("embedding length %d" % len(embeddings))
    print("rev_vocab length %d" % len(rev_vocab))
    print("vocab length %d" % len(vocab))
    model = du.choose_model(embeddings=embeddings)

    with tf.Session() as sess:
        du.restore_model(session=sess, run_id=FLAGS.run_id)
        answers = generate_answers(sess, model, dataset, rev_vocab=rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))

    evaluate.main(FLAGS.dev_path, 'dev-prediction.json')


if __name__ == "__main__":
    parse_args()
    tf.app.run()
