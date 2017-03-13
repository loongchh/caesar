import logging
import time

import numpy as np
import tensorflow as tf

import qa_data_util as du
import evaluate
import parse_args
from util import Progbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS


from match_lstm_boundry import MatchLstmBoundryModel

logger = logging.getLogger("hw4")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def choose_model(embeddings, debug_shape=False):
    if FLAGS.model.lower() == "match_lstm":
        from match_lstm import MatchLstmModel
        model = MatchLstmModel(embeddings, debug_shape)
    elif FLAGS.model.lower() == "match_lstm_boundry":
        from match_lstm_boundry import MatchLstmBoundryModel
        model = MatchLstmBoundryModel(embeddings, debug_shape)
    else:
        model = None

    return model


def train_epoch(train_data, model, session, losses, grad_norms):
    num_train_batches = int(len(train_data['q'])/FLAGS.batch_size)
    prog = Progbar(target=num_train_batches)

    for i in range(num_train_batches):
        if i >= FLAGS.train_batch >= 0:
            break
        data_batch = du.get_batch(train_data, i)
        grad_norm, loss, pred = model.train_on_batch(sess=session, data_batch=data_batch)
        losses.append(loss)
        grad_norms["Q_LSTM/RNN/LSTMCell"].append(grad_norm[0])
        grad_norms["P_LSTM/RNN/LSTMCell"].append(grad_norm[1])
        grad_norms["Match_LSTM_fwd/LSTMCell"].append(grad_norm[2])
        grad_norms["Match_LSTM_rev/LSTMCell"].append(grad_norm[3])
        grad_norms["ANSWER_POINTER/LSTMCell"].append(grad_norm[4])
        grad_norms["REST"].append(grad_norm[5])
        prog.update(i+1, [("grad_norm", np.average(grad_norm)), ("train loss", loss)])
    print ""
    return grad_norms, losses


# def evaluate_single(document, question, ground_truth_span, predicted_span, rev_vocab):
#         f1 = 0
#         em = False
#
#         ## Reverse the indices if start is greater than end, SHOULDN'T Happen
#         if predicted_span[0] > predicted_span[1]:
#             a = predicted_span[0]
#             predicted_span[0]=predicted_span[1]
#             predicted_span[1] = a
#
#         ground_truth_tokens = [rev_vocab[int(token_id)] for index, token_id in enumerate(document)
#                                 if int(ground_truth_span[0]) <= int(index) <= int(ground_truth_span[1])]
#
#         predicted_tokens = [rev_vocab[int(token_id)] for index, token_id in enumerate(document)
#                                 if int(predicted_span[0]) <= int(index) <= int(predicted_span[1])]
#
#         predicted = " ".join(predicted_tokens)
#         ground_truth = " ".join(ground_truth_tokens)
#         if em:
#             print predicted, document, question
#         f1 = evaluate.f1_score(predicted, ground_truth)
#         em = evaluate.exact_match_score(predicted, ground_truth)
#         return f1, em

def evaluate_single(document, ground_truth, predicted, rev_vocab, print_answer_text):
        f1 = 0
        em = False

        ground_truth_tokens = [rev_vocab[document[index]] for index in ground_truth]
        predicted_tokens = [rev_vocab[document[index]] for index in predicted if index < FLAGS.max_document_size]

        predicted_text = " ".join(predicted_tokens)
        ground_truth_text = " ".join(ground_truth_tokens)

        f1 = evaluate.f1_score(predicted_text, ground_truth_text)
        em = evaluate.exact_match_score(predicted_text, ground_truth_text)
        if em:
            logger.info("--------Match!!--------------")
            logger.info("Ground truth: {}".format(ground_truth_text))
            logger.info("Predicted Answer: {}".format(predicted_text))
            logger.info("-----------------------------")
        elif print_answer_text:
            logger.info("Ground truth: {}".format(ground_truth_text))
            logger.info("Predicted Answer: {}".format(predicted_text))
        return f1, em


def evaluate_batch(data_batch, predicted_batch, rev_vocab, print_answer_text):
    f1_sum = 0.
    em_sum = 0.
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
        f1_sum += f1
        em_sum += 1. if em else 0.


    return f1_sum/len(predicted_batch), em_sum


def evaluate_epoch(val_data, model, session, rev_vocab, print_answer_text):
    logger.info("Dev Evaluation")
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
        data_batch = du.get_batch(val_data, i)
        pred = model.predict_on_batch(sess=session, data_batch=data_batch)
        f1, em = evaluate_batch(
            data_batch=data_batch,
            predicted_batch=pred,
            rev_vocab=rev_vocab,
            print_answer_text=(print_answer_text)
        )
        f1_sum += f1
        em_sum += em
        # prog.update(i+1, [("Avg F1", f1)])
    print ""
    logger.info("Evaluation: Avg F1 Score: {}. Total EM Score: {} out of {}".format(f1_sum/batch_size, em_sum, data_size))
    return f1_sum/batch_size, em_sum/batch_size


def train():
    logger.info("----------------Training model-----------------------------")
    vocab,rev_vocab = du.initialize_vocab()

    embeddings = du.load_embeddings()
    train_data = du.load_dataset(type = "train")
    val_data = du.load_dataset(type = "val")

    with tf.Graph().as_default():

        logger.info("Building model...",)
        start = time.time()
        model = choose_model(embeddings=embeddings, debug_shape=True)
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            # session = tf_debug.LocalCLIDebugWrapperSession(session)
            # session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
            session.run(init)

            losses = []
            grad_norms = {"Q_LSTM/RNN/LSTMCell":[],
                             "P_LSTM/RNN/LSTMCell":[],
                             "Match_LSTM_fwd/LSTMCell":[],
                             "Match_LSTM_rev/LSTMCell":[],
                             "ANSWER_POINTER/LSTMCell":[],
                             "REST":[]
                             }
            for epoch in range(FLAGS.epochs):

                # run_metadata = tf.RunMetadata()
                # train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)

                logger.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
                ### Training
                grad_norms, losses = train_epoch(train_data, model, session,losses, grad_norms)
                ### Evaluation
                f1, em = evaluate_epoch(val_data, model, session, rev_vocab, print_answer_text=(FLAGS.print_text == 1))

                ### Checkpoint model
            make_prediction_plot(losses, grad_norms)

            # train_writer.close()

    logger.info("Model did not crash!")
    logger.info("Passed!")


def make_prediction_plot(losses, grad_norms):
    losses = np.array(losses)
    plt.subplot(7, 1, 1)
    plt.title("Loss")
    plt.plot(np.arange(losses.size), losses.flatten(), label="Loss")
    plt.ylabel("Loss")

    for i,key in enumerate(grad_norms):
        norm = np.array(grad_norms[key])
        plt.subplot(7, 1, 2+i)
        plt.plot(np.arange(norm.size), norm.flatten(), label="Gradients")
        plt.ylabel("Gradients-{}".format(key))
        plt.xlabel("Minibatch")

    output_path = "../plots/train.png"
    plt.savefig(output_path)


def log_total_parametes():

    total_parameters = 0
    for variable in tf.trainable_variables():
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
    logger.info("total parameters: {}".format(total_parameters))


def debug_shape():
    embeddings = du.load_embeddings()
    val_data = du.load_dataset(type = "val")
    vocab,rev_vocab = du.initialize_vocab()
    logger.info("----------------Debugging Tensor Shapes-----------------------------")
    with tf.Graph().as_default():

        logger.info("Building model for Debugging Shape...")
        start = time.time()
        model = choose_model(embeddings=embeddings, debug_shape=True)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        log_total_parametes()

        with tf.Session() as session:
            session.run(init)
            model.debug_shape(
                session,
                data_batch=du.get_batch(val_data,0)
            )
    logger.info("")


if __name__ == "__main__":
    parse_args.parse_args()
    if FLAGS.debug_shape == 1:
        debug_shape()
    train()