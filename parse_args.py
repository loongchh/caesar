import tensorflow as tf

def parse_args():
    # model
    tf.app.flags.DEFINE_string("model", "coattention_without_summary", "coattention/match_lstm/match_lstm_boundry/coattention_bilstm/seq2seq")
    tf.app.flags.DEFINE_string("run_id","", "model run id, eg. 2017-03-15-01-51-39")

    # Hyper Parameters
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
    tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
    tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
    tf.app.flags.DEFINE_integer("batch_size", 40, "Batch size to use during training.")
    tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
    tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
    tf.app.flags.DEFINE_string("tokenizer", "CORE-NLP", "NLTK/CORE-NLP")
    tf.app.flags.DEFINE_string("glove_crawl_size", "6B", "Crawl size of embeddings")
    tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
    tf.app.flags.DEFINE_integer("min_document_size", 0, "")
    tf.app.flags.DEFINE_integer("max_document_size", 600, "")
    tf.app.flags.DEFINE_integer("min_question_size", 0, "")
    tf.app.flags.DEFINE_integer("max_question_size", 41, "")
    tf.app.flags.DEFINE_integer("min_answer_size", 0, "")
    tf.app.flags.DEFINE_integer("max_answer_size", 20, "")
    tf.app.flags.DEFINE_bool("embedding_trainable", False, "Allow training of embedding vectors")

    # Model Specific Parameters
    # Coattention
    tf.app.flags.DEFINE_integer("max_summary_size", 600, "Truncate the document to specific length. MUST BE EVEN.")
    tf.app.flags.DEFINE_string("pool_type", "max", "Pooling mechanism used to summarize each sentence.")

    # Directories
    tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
    tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
    tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
    tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
    tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
    tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

    # Training, Debugging, QA Answer
    tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
    tf.app.flags.DEFINE_bool("debug", False, "Debug mode")
    tf.app.flags.DEFINE_bool("test_summary", False, "Testing answers retained in summarization.")
    tf.app.flags.DEFINE_integer("train_batch", -1, "No of batches used in training. Set -1 to train on all.")
    tf.app.flags.DEFINE_integer("val_batch", -1, "No of batches used in validaton. Set -1 to validate on all.")
    tf.app.flags.DEFINE_integer("print_text", 1, "Print predicted text after every n epochs")

    tf.app.flags.DEFINE_string("comment", "", "Comment that will be printed in the end, put some")
    tf.app.flags.DEFINE_integer("cluster_mode", 0, "whether the training is on gpu cluster")
    tf.app.flags.DEFINE_bool("codalab", False, "whether the execution is on codalab")
