import tensorflow as tf

def parse_args():
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
    tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
    tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
    tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")

    tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
    tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
    tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
    tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
    tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
    tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
    tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
    tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
    tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
    tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
    tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")

    tf.app.flags.DEFINE_integer("debug_shape", 0, "Debug shapes of Tensor")
    tf.app.flags.DEFINE_integer("train_batch", -1, "No of batches used in training. Set -1 to train on all.")
    tf.app.flags.DEFINE_integer("val_batch", -1, "No of batches used in validaton. Set -1 to validate on all.")
    tf.app.flags.DEFINE_integer("print_text", 0, "Print predicted text after every n epochs")

    tf.app.flags.DEFINE_string("model","coattention", "coattention/match_lstm/match_lstm_boundry")
    tf.app.flags.DEFINE_integer("coattention_pooling_size",11, "")

    tf.app.flags.DEFINE_integer("min_document_size", 0, "")
    tf.app.flags.DEFINE_integer("max_document_size", 300, "")

    tf.app.flags.DEFINE_integer("min_question_size", 0, "")
    tf.app.flags.DEFINE_integer("max_question_size", 25, "")

    tf.app.flags.DEFINE_integer("min_answer_size", 0, "")
    tf.app.flags.DEFINE_integer("max_answer_size", 6, "")

    tf.app.flags.DEFINE_string("comment", "", "Comment that will be printed in the end, put some")

    tf.app.flags.DEFINE_integer("maxout_size", 32, "Size of maxout in Highway Maxout Network.")
    tf.app.flags.DEFINE_integer("max_decode_steps", 4, "Size of decoding steps.")


