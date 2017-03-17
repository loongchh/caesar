import logging
import time

import numpy as np
import tensorflow as tf

from qa_data_util import *
import evaluate
import parse_args
from util import Progbar
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def train_epoch(train_data, model, session, losses, grad_norms):
    num_train_batches = int(len(train_data['q']) / FLAGS.batch_size)
    prog = Progbar(target=num_train_batches)
    permutation = np.random.permutation(num_train_batches*FLAGS.batch_size)
    for i in range(num_train_batches):
        if i >= FLAGS.train_batch >= 0:
            break
        data_batch = get_batch(train_data, i, permutation=permutation)
        (grad_norm, loss) = model.train_on_batch(sess=session, data_batch=data_batch)
        losses.append(loss)
        for j,grad in enumerate(grad_norm):
            grad_norms[j].append(grad)
        prog.update(i+1, [("grad_norm",np.sum(grad_norm)), ("loss", loss)])
    print ""
    return grad_norms, losses


def evaluate_single(document, ground_truth, predicted, rev_vocab, print_answer_text):
    f1 = 0
    em = False

    ground_truth_tokens = [rev_vocab[document[index]] for index in ground_truth]
    predicted_tokens = [rev_vocab[document[index]] for index in predicted if index < FLAGS.max_document_size]

    predicted_text = " ".join(predicted_tokens)
    ground_truth_text = " ".join(ground_truth_tokens)

    f1 = evaluate.f1_score(predicted_text, ground_truth_text)
    em = evaluate.exact_match_score(predicted_text, ground_truth_text)
    # if em:
    #     logger.info("--------Match!!--------------")
    #     logger.info("Ground truth: {}".format(ground_truth_text))
    #     logger.info("Predicted Answer: {}".format(predicted_text))
    #     logger.info("-----------------------------")
    if print_answer_text:
        logger.info("Ground truth: {}".format(ground_truth_text))
        logger.info("Predicted Answer: {}".format(predicted_text))
    return f1, em


def evaluate_batch(data_batch, predicted_batch, rev_vocab, print_answer_text):
    f1_sum_batch = 0.
    em_sum_batch = 0.
    for i in range(len(data_batch['q'])):
        q = data_batch['q']
        c = data_batch['c'][i]
        gt = data_batch['gt'][i]
        pred = predicted_batch[i]

        f1, em = evaluate_single(
            document=c,
            ground_truth=gt,
            predicted=pred,
            rev_vocab=rev_vocab,
            print_answer_text=print_answer_text and (i % 5 ==1)
        )
        f1_sum_batch += f1
        em_sum_batch += 1. if em else 0.

    return f1_sum_batch, em_sum_batch


def evaluate_epoch(val_data, model, session, rev_vocab, print_answer_text):
    logger.info("=============== Evaluation ===============")
    f1_sum = 0
    em_sum = 0
    batch_size = FLAGS.batch_size

    data_size = len(val_data['q'])
    num_val_batches = int(data_size/batch_size)
    data_size = num_val_batches * batch_size
    # prog = Progbar(target= num_val_batches)
    for i in range(num_val_batches):
        if i >= FLAGS.val_batch >= 0:
            break
        data_batch = get_batch(val_data, i)
        pred = model.predict_on_batch(sess=session, data_batch=data_batch)
        f1_sum_batch, em_sum_batch = evaluate_batch(
            data_batch=data_batch,
            predicted_batch=pred,
            rev_vocab=rev_vocab,
            print_answer_text=(print_answer_text)
        )
        f1_sum += f1_sum_batch
        em_sum += em_sum_batch
        # prog.update(i+1, [("Avg F1", f1)])
    print ""
    logger.info("F1 Score: {}. EM Score: {} out of {}".format(f1_sum / data_size, em_sum, data_size))
    return f1_sum/data_size, em_sum


def train():
    run_id = str(datetime.now()).split(".")[0].replace(' ','_').replace(':','_').replace('-','_')
    logger.info("=============== Training model - {} ===============".format(run_id))
    vocab,rev_vocab = initialize_vocab()

    embeddings = load_embeddings()
    train_data = load_dataset(type="train")
    val_data = load_dataset(type="val")

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = choose_model(embeddings=embeddings)
        logger.info("Took %.2f seconds.", time.time() - start)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # TODO: Play more with TFDG Debugger
            # session = tf_debug.LocalCLIDebugWrapperSession(session)
            # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # TODO: tensorboard summary writer
            # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
            session.run(init)

            F1s, EMs, losses = [], [], []
            grad_norms = [[] for i,j in enumerate(tf.trainable_variables())] # grad_norm array for all the variables


            for epoch in range(FLAGS.epochs):

                # TODO: tensorboard summary writer
                # run_metadata = tf.RunMetadata()
                # train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)

                logger.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                # Training
                grad_norms, losses = train_epoch(train_data, model, session,losses, grad_norms)
                # Evaluation
                f1, em = evaluate_epoch(val_data, model, session, rev_vocab, print_answer_text=(FLAGS.print_text == 1))
                F1s.append(f1)
                EMs.append(em)
                logger.info(F1s)
                logger.info(EMs)
                # Checkpoint model
                if f1 == max(F1s):
                    logger.info("New best model! Saving...")
                    checkpoint_model(session, run_id)

            make_training_plots(losses, grad_norms, F1s, EMs, run_id)

            # train_writer.close()

    logger.info("Training finished.")
    logger.info(vars(FLAGS))


def make_training_plots(losses, grad_norms, F1s, EMs, run_id):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        now = datetime.utcnow()
        with PdfPages("../plots/{}-{}.pdf".format(FLAGS.model, run_id)) as pdf:
            plt.clf()
            # -----------------------
            F1s = np.array(F1s)
            plt.figure()
            plt.title("F1 Score")
            plt.plot(np.arange(F1s.size), F1s.flatten(), label="F1 Score")
            plt.ylabel("F1 Score")
            pdf.savefig()
            plt.close()
            # -----------------------
            EMs = np.array(EMs)
            plt.figure()
            plt.title("EM Count (out of 1360)")
            plt.plot(np.arange(EMs.size), EMs.flatten(), label="EM Count")
            plt.ylabel("EM Count")
            pdf.savefig()
            plt.close()
            # -----------------------
            losses = np.array(losses)
            plt.figure()
            plt.title("Loss")
            plt.plot(np.arange(losses.size), losses.flatten(), label="Loss")
            plt.ylabel("Loss")
            pdf.savefig()
            plt.close()

            for i,v in enumerate(tf.trainable_variables()):
                norm = np.array(grad_norms[i])
                plt.figure()
                plt.title(v.name)
                plt.plot(np.arange(norm.size), norm.flatten(), label=v.name)
                plt.ylabel(v.name)
                pdf.savefig()
                plt.close()

    except ImportError:
        pass

def log_total_parametes():
    logger.debug("Trainable variables:")
    total_parameters = 0
    for variable in tf.trainable_variables():
        logger.debug(variable.name)
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        # print(variable_parametes)
        total_parameters += variable_parametes
    logger.debug("Total parameters in model: {}".format(total_parameters))


def debug():
    embeddings = load_embeddings()
    val_data = load_dataset(type = "val", debug=True)
    vocab,rev_vocab = initialize_vocab()
    logger.debug("==================== Debug ====================")
    with tf.Graph().as_default():

        logger.debug("Building model...")
        start = time.time()
        model = choose_model(embeddings=embeddings, debug=True)
        logger.info("Took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        log_total_parametes()

        with tf.Session() as session:
            session.run(init)
            model.debug(
                session,
                data_batch=get_batch(val_data,0)
            )

def test_summary_size():
    logger.info("=============== Testing summary size ===============")
    vocab,rev_vocab = initialize_vocab()

    embeddings = load_embeddings()
    val_data = load_dataset(type="val")

    with tf.Graph().as_default():
        logger.debug("Building model...",)
        start = time.time()
        model = choose_model(embeddings=embeddings)
        logger.debug("Took %.2f seconds.", time.time() - start)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            # TODO: Play more with TFDG Debugger
            # session = tf_debug.LocalCLIDebugWrapperSession(session)
            # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # TODO: tensorboard summary writer
            # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
            session.run(init)

            successes_per_epoch = []
            for epoch in range(FLAGS.epochs):

                # TODO: tensorboard summary writer
                # run_metadata = tf.RunMetadata()
                # train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)

                logger.debug("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                # Training
                successes_per_epoch.append(summary_success_epoch(val_data, model, session))
                

    def summary_success_epoch(train_data, model, session):
        num_train_batches = int(len(train_data['q']) / FLAGS.batch_size)
        prog = Progbar(target=num_train_batches)
        permutation = np.random.permutation(num_train_batches*FLAGS.batch_size)
        successes = []
        for i in range(num_train_batches):
            if i >= FLAGS.train_batch >= 0:
                break
            data_batch = get_batch(train_data, i, permutation=permutation)
            successes.append(model.summary_success(sess=session, data_batch=data_batch))
            prog.update(i+1, [("successes", sum(successes))])

        logger.debug("Summarization: %d out of %d answers are retained", sum(successes), int(len(train_data['q'])))
        return sum(successes)

if __name__ == "__main__":
    parse_args.parse_args()
    if FLAGS.debug == 1:
        debug()
        if FLAGS.max_summary_size < FLAGS.max_document_size:
            test_summary_size()
    train()

