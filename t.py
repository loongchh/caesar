import logging, time
import tensorflow as tf
import numpy as np
import qa_data_util as du
import evaluate
import parse_args
from util import Progbar
FLAGS = tf.app.flags.FLAGS

from coattention_model import CoattentionModel
from baseline_model import BaselineModel

logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def evaluate_single(document, question, ground_truth_span, predicted_span, rev_vocab):
        f1 = 0
        em = False

        ## Reverse the indices if start is greater than end, SHOULDN'T Happen
        if predicted_span[0] > predicted_span[1]:
            a = predicted_span[0]
            predicted_span[0]=predicted_span[1]
            predicted_span[1] = a

        ground_truth_tokens = [rev_vocab[int(token_id)] for index, token_id in enumerate(document)
                                if int(ground_truth_span[0]) <= int(index) <= int(ground_truth_span[1])]

        predicted_tokens = [rev_vocab[int(token_id)] for index, token_id in enumerate(document)
                                if int(predicted_span[0]) <= int(index) <= int(predicted_span[1])]

        predicted = " ".join(predicted_tokens)
        ground_truth = " ".join(ground_truth_tokens)
        if em:
            print predicted, document, question
        f1 = evaluate.f1_score(predicted, ground_truth)
        em = evaluate.exact_match_score(predicted, ground_truth)
        return f1, em


def train_epoch(train_data, model, session):
    num_train_batches = int(len(train_data['q'])/FLAGS.batch_size)
    prog = Progbar(target=1 + num_train_batches)
    for i in range(num_train_batches):
        start = i*FLAGS.batch_size
        end = (i+1)*FLAGS.batch_size
        loss, pred = model.train_on_batch(
            session,
            train_data['q'][start: end],
            train_data['c'][start: end],
            train_data['s'][start: end],
        )
        prog.update(i+1, [("train loss", loss)])

def evaluate_epoch(val_data, model, session, rev_vocab):
    logger.info("Dev Evaluation")
    f1_sum = 0
    em_sum = 0
    batch_size = FLAGS.batch_size
    data_size = len(val_data['q'])
    num_val_batches = int(data_size/batch_size)
    data_size = num_val_batches * batch_size
    prog = Progbar(target=1 + num_val_batches)
    for i in range(num_val_batches):
        start = i*FLAGS.batch_size
        end = (i+1)*FLAGS.batch_size
        pred = model.predict_on_batch(
            session,
            val_data['q'][start: end],
            val_data['c'][start: end]
        )
        for j in range(start, end):
            if int(val_data['s'][j][0]) == int(val_data['s'][j][1]) and int(val_data['s'][j][1]) == FLAGS.max_document_size -1:
                # logger.info( "{} skipped".format(j))
                continue
            f1, em = evaluate_single(val_data['c'][j], val_data['q'][j], val_data['s'][j], pred[j%FLAGS.batch_size],rev_vocab)
            f1_sum += f1
            em_sum += 1. if em else 0.
        prog.update(i+1, [("F1", f1_sum/((i+1)*batch_size))])

    logger.info("Evaluation: F1 Score: {}. EM Score: {}".format(f1_sum/data_size, em_sum/data_size))
    return f1_sum/data_size, em_sum/data_size


def train():
    vocab,rev_vocab = du.initialize_vocab()
    # print vocab['<pad>']

    embeddings = du.load_embeddings()
    train_data = du.load_dataset(type = "train")
    val_data = du.load_dataset(type = "val")



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

            # for epoch in range(1):
            for epoch in range(FLAGS.epochs):

                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
                logger.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                ### Training
                train_epoch(train_data, model, session)

                ### Evaluation
                f1, em = evaluate_epoch(val_data, model, session, rev_vocab)

                ### Checkpoint model
            train_writer.close()

    logger.info("Model did not crash!")
    logger.info("Passed!")


def debug_shape():
    embeddings = du.load_embeddings()
    val_data = du.load_dataset(type = "val")
    vocab,rev_vocab = du.initialize_vocab()
    logger.info("----------------------------------------------------------")
    with tf.Graph().as_default():

        logger.info("Building model for Debugging Shape...")
        start = time.time()
        # model = CoattentionModel(embeddings, debug_shape=True)
        model = BaselineModel(embeddings, debug_shape=True)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            model.debug_shape(
                session,
                question_batch=val_data['q'][0:FLAGS.batch_size],
                document_batch=val_data['c'][0:FLAGS.batch_size],
                span_batch=val_data['s'][0:FLAGS.batch_size]
            )
    logger.info("----------------------------------------------------------")



if __name__ == "__main__":
    parse_args.parse_args()
    # debug_shape()
    train()
