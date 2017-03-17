import tensorflow as tf

def parse_args():
    # model
    tf.app.flags.DEFINE_string("model","coattention", "coattention/match_lstm/match_lstm_boundry")
    tf.app.flags.DEFINE_string("run-id","model1", "model run id, eg. 2017-03-15-01-51-39")

    # Hyper Parameters
    tf.app.flags.DEFINE_float("learning-rate", 0.001, "Learning rate.")
    tf.app.flags.DEFINE_float("max-gradient-norm", 5.0, "Clip gradients to this norm.")
    tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
    tf.app.flags.DEFINE_integer("batch-size", 40, "Batch size to use during training.")
    tf.app.flags.DEFINE_integer("state-size", 200, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("embedding-size", 100, "Size of the pretrained vocabulary.")
    tf.app.flags.DEFINE_string("glove-crawl-size", "6B", "Crawl size of embeddings")
    # tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
    tf.app.flags.DEFINE_integer("min-document-size", 0, "")
    tf.app.flags.DEFINE_integer("max-document-size", 300, "")
    tf.app.flags.DEFINE_integer("min-question-size", 0, "")
    tf.app.flags.DEFINE_integer("max-question-size", 25, "")
    tf.app.flags.DEFINE_integer("min-answer-size", 0, "")
    tf.app.flags.DEFINE_integer("max-answer-size", 6, "")

    # Model Specific Parameters
    # Coattention
    tf.app.flags.DEFINE_integer("max-summary-size", 300, "Truncate the document to specific length.")
    tf.app.flags.DEFINE_string("pool-type", "max", "Pooling mechanism used to summarize each sentence.")

    # Directories
    tf.app.flags.DEFINE_string("vocab-path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
    tf.app.flags.DEFINE_string("embed-path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
    tf.app.flags.DEFINE_string("data-dir", "data/squad", "SQuAD directory (default ./data/squad)")
    tf.app.flags.DEFINE_string("train-dir", "train", "Training directory to save the model parameters (default: ./train).")
    tf.app.flags.DEFINE_string("log-dir", "log", "Path to store log and flag files (default: ./log)")
    tf.app.flags.DEFINE_string("dev-path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

    # Training, Debugging, QA Answer
    tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
    tf.app.flags.DEFINE_bool("debug", False, "Debug mode")
    tf.app.flags.DEFINE_integer("train-batch", -1, "No of batches used in training. Set -1 to train on all.")
    tf.app.flags.DEFINE_integer("val-batch", -1, "No of batches used in validaton. Set -1 to validate on all.")
    tf.app.flags.DEFINE_integer("print-text", 1, "Print predicted text after every n epochs")
    tf.app.flags.DEFINE_integer("cluster-mode", 0, "Print predicted text after every n epochs")
    # tf.app.flags.DEFINE_integer("print-every", 1, "How many iterations to do per print.")
    # tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")

    tf.app.flags.DEFINE_string("comment", "", "Comment that will be printed in the end, put some")
