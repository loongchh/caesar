#!/usr/bin/env bash
# Downloads and preprocesses data into ./data
# Get directory containing this script

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR

pip install -r $CODE_DIR/requirements.txt

# download punkt, perluniprops
python2 -m nltk.downloader punkt

# SQuAD preprocess is in charge of downloading
# and formatting the data to be consumed later
DATA_DIR=data
mkdir -p $DATA_DIR
rm -rf $DATA_DIR
python2 $CODE_DIR/preprocessing/squad_preprocess.py

# Download distributed word representations
python2 $CODE_DIR/preprocessing/dwr.py

# Data processing for TensorFlow
python2 $CODE_DIR/qa_data.py --glove_dim 100
