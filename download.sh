#!/usr/bin/env bash

DOWNLOAD_PATH=${1-'~/data/TMLGA'}

mkdir -p DOWNLOAD_PATH
mkdir DOWNLOAD_PATH/word_embeddings

# download pre-trained spacy models
python -m spacy download en_core_web_md

# download pre-trained GLoVe word embeddings
wget -P DOWNLOAD_PATH/word_embeddings http://nlp.stanford.edu/data/glove.840B.300d.zip 
unzip DOWNLOAD_PATH/word_embeddings/glove.840B.300d.zip

# download CHARADES features
wget -P DOWNLOAD_PATH https://zenodo.org/record/3590426/files/i3d_charades_sta.zip
unzip DOWNLOAD_PATH/i3d_charades_sta.zip -d i3d_charades_sta   

# download CHARADES pre-trained model
mkdir -p checkpoints/charades_sta
wget -P ./checkpoints/charades_sta/ https://zenodo.org/record/3590426/files/model_charades_sta
