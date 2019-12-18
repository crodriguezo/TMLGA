import os
import json
import time
import math
import torch
import pickle
import numpy as np
from random import shuffle

from utils.vocab import Vocab
from utils.sentence import get_embedding_matrix

from torch.utils.data import Dataset

from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


class CHARADES_STA(Dataset):

    def __init__(self, features_path,
                       ann_file_path,
                       embeddings_path,
                       min_count,
                       train_max_length,
                       test_max_length):

        self.feature_path = features_path
        self.ann_file_path = ann_file_path
        self.is_training = 'train' in ann_file_path
        print(self.is_training)

        print('loading annotations into memory...', end=" ")
        tic = time.time()
        self.dataset = json.load(open(ann_file_path, 'r'))

        # self.glove = np.load(vocab_glove, allow_pickle=True).item()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))

        self.min_count = min_count
        self.train_max_length = train_max_length
        self.test_max_length = test_max_length

        vocab_file_name = f'charades_vocab_{min_count}_{train_max_length}.pickle'

        self.vocab_file_path = vocab_file_name
        self.create_vocab()

        embeddings_file_name = f'charades_embeddings_{min_count}_{train_max_length}.pth'
        self.embeddings_file_path = embeddings_file_name
        self.get_embedding_matrix(embeddings_path)

        self.createIndex()
        self.ids   = list(self.anns.keys())
        self.epsilon = 1E-10

    def create_vocab(self):
        print(self.vocab_file_path, os.path.exists(self.vocab_file_path))
        if self.is_training:
            if not os.path.exists(self.vocab_file_path):
                print("Creating vocab")
                self.vocab = Vocab(
                    add_bos=False,
                    add_eos=False,
                    add_padding=False,
                    min_count=self.min_count)

                for example in self.dataset:
                    self.vocab.add_tokenized_sentence(example['tokens'][:self.train_max_length])

                self.vocab.finish()

                with open(self.vocab_file_path, 'wb') as f:
                    pickle.dump(self.vocab, f)
            else:
                with open(self.vocab_file_path, 'rb') as f:
                    self.vocab = pickle.load(f)

        else:
            print("Cargando vocab")
            with open(self.vocab_file_path, 'rb') as f:
                self.vocab = pickle.load(f)


    def get_embedding_matrix(self, embeddings_path):
        '''
        Gets you a torch tensor with the embeddings
        in the indices given by self.vocab.

        Unknown (unseen) words are each mapped to a random,
        different vector.


        :param embeddings_path:
        :return:
        '''
        if self.is_training and not os.path.exists(self.embeddings_file_path):
            tic = time.time()

            print('loading embeddings into memory...', end=" ")

            if 'glove' in embeddings_path.lower():
                tmp_file = get_tmpfile("test_word2vec.txt")
                _ = glove2word2vec(embeddings_path, tmp_file)
                embeddings = KeyedVectors.load_word2vec_format(tmp_file)
            else:
                embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

            print('Done (t={:0.2f}s)'.format(time.time() - tic))

            embedding_matrix = get_embedding_matrix(embeddings, self.vocab)

            with open(self.embeddings_file_path, 'wb') as f:
                torch.save(embedding_matrix, f)

        else:
            with open(self.embeddings_file_path, 'rb') as f:
                embedding_matrix  = torch.load(f)

        self.embedding_matrix = embedding_matrix


    def createIndex(self):
        print("Creating index..", end=" ")
        anns = {}
        size = int(round(len(self.dataset) * 1.))
        counter = 0
        for row in self.dataset[:size]:
            if float(row['feature_start']) > float(row['feature_end']):
                print(row)
                continue

            if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                row['feature_end'] = float(row['number_features'])-1

            row['augmentation'] = 0
            anns[counter] = row
            counter+=1

        self.anns = anns
        print(" Ok! {}".format(len(anns.keys())))

    def __getitem__(self, index):
        ann = self.anns[index]

        i3dfeat = "{}/{}.npy".format(self.feature_path, ann['video'])
        i3dfeat = np.load(i3dfeat)
        i3dfeat = np.squeeze(i3dfeat)
        i3dfeat = torch.from_numpy(i3dfeat)
        feat_length = i3dfeat.shape[0]

        if self.is_training:
            raw_tokens = ann['tokens'][:self.train_max_length]
        else:
            raw_tokens = ann['tokens'][:self.test_max_length]

        indices = self.vocab.tokens2indices(raw_tokens)
        tokens = [self.embedding_matrix[index] for index in indices]
        tokens = torch.stack(tokens)

        localization = np.zeros(feat_length, dtype=np.float32)
        start = math.floor(ann['feature_start'])
        end   = math.floor(ann['feature_end'])
        time_start = ann['time_start']
        time_end = ann['time_end']

        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end   = np.ones(feat_length, dtype=np.float32) * self.epsilon
        y = (1 - (feat_length-3) * self.epsilon - 0.5)/ 2

        if start > 0:
            loc_start[start - 1] = y
        if start < feat_length-1:
            loc_start[start + 1] = y
        loc_start[start] = 0.5

        if end > 0:
            loc_end[end - 1] = y
        if end < feat_length-1:
            loc_end[end + 1] = y
        loc_end[end] = 0.5

        y = 1.0
        localization[start:end] = y

        return index, i3dfeat, tokens, torch.from_numpy(loc_start), torch.from_numpy(loc_end), torch.from_numpy(localization),\
               time_start, time_end, ann['number_frames']/ann['number_features'], ann['fps']

    def __len__(self):
        return len(self.ids)
