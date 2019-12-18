#!/usr/bin/env python
# -*-coding: utf8 -*-
import random

random.seed(42)

UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'


class VocabItem:

    def __init__(self, string, hash=None):
        self.string = string
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding
        self.hash = hash

    def __str__(self):
        return 'VocabItem({})'.format(self.string)

    def __repr__(self):
        return self.__str__()



class Vocab:

    def __init__(self, min_count=1, add_padding=False, add_bos=False,
                 add_eos=False, unk=None, lowercase=False, max_size=None,
                 no_unk=False):
        """
        :param sentences:
        :param token_function:
        :param min_count:
        :param add_padding:
        :param add_bos:
        :param add_eos:
        :param unk:
        """

        if min_count and max_size:
            raise ValueError()

        self.vocab_items = []
        self.vocab_hash = {}
        self.word_count = 0
        self.lowercase = lowercase
        self.special_tokens = []
        self.min_count = min_count
        self.add_padding =  add_padding
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.unk = unk
        self.max_size = max_size
        self.no_unk = no_unk

        self.UNK = None
        self.PAD = None
        self.BOS = None
        self.EOS = None

        self.index2token = []
        self.token2index = {}

        self.finished = False

    def add_tokenized_sentence(self, tokens):
        if self.finished:
            raise RuntimeError('Vocabulary is finished')

        for token in tokens:
            real_token = token.lower() if self.lowercase else token
            if real_token not in self.vocab_hash:
                self.vocab_hash[real_token] = len(self.vocab_items)
                self.vocab_items.append(VocabItem(real_token))

            self.vocab_items[self.vocab_hash[real_token]].count += 1
            self.word_count += 1

    def finish(self):

        token2index = self.token2index
        index2token = self.index2token

        tmp = []

        if not self.no_unk:

            if self.unk:
                self.UNK = VocabItem(self.unk, hash=0)
                self.UNK.count = self.vocab_items[self.vocab_hash[self.unk]].count
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                for token in self.vocab_items:
                    if token.string != self.unk:
                        tmp.append(token)

            else:
                self.UNK = VocabItem(UNK, hash=0)
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                if self.min_count is not None:
                    for token in self.vocab_items:
                        if token.count < self.min_count:
                            self.UNK.count += token.count
                        else:
                            tmp.append(token)

                    tmp.sort(key=lambda token: token.count, reverse=True)

                elif self.max_size:
                    for token in self.vocab_items:
                        tmp.append(token)

                    tmp.sort(key=lambda token: token.count, reverse=True)

                    if self.max_size <= len(tmp):
                        for token in tmp[self.max_size:]:
                            self.UNK.count += token.count

                        tmp = tmp[:self.max_size]

                else:
                    raise ValueError()

        else:
            for token in self.vocab_items:
                tmp.append(token)

        if self.add_bos:
            self.BOS = VocabItem(BOS)
            tmp.append(self.BOS)
            self.special_tokens.append(self.BOS)

        if self.add_eos:
            self.EOS = VocabItem(EOS)
            tmp.append(self.EOS)
            self.special_tokens.append(self.EOS)

        if self.add_padding:
            self.PAD = VocabItem(PAD)
            tmp.append(self.PAD)
            self.special_tokens.append(self.PAD)

        index2token += tmp

        # Update vocab_hash

        for i, token in enumerate(self.index2token):
            token2index[token.string] = i
            token.hash = i

        self.index2token = index2token
        self.token2index = token2index

        if not self.no_unk:
            print('Unknown vocab size:', self.UNK.count)

        print('Vocab size: %d' % len(self))

        self.finished = True


    def __getitem__(self, i):
        return self.index2token[i]

    def __len__(self):
        return len(self.index2token)

    def __iter__(self):
        return iter(self.index2token)

    def __contains__(self, key):
        return key in self.token2index


    def tokens2indices(self, tokens, add_bos=False, add_eos=False, oovs=None):

        string_seq = []

        if add_bos:
            string_seq.append(self.BOS.hash)
        for item in tokens:
            processed_token = item.lower() if self.lowercase else item
            if oovs is not None:
                if processed_token in self.token2index:
                    string_seq.append(self.token2index[processed_token])
                else:
                    if processed_token not in oovs:
                        oovs.append(processed_token)
                    oov_id = len(self) + oovs.index(processed_token)
                    string_seq.append(oov_id)
            else:
                if self.no_unk:
                    string_seq.append(self.token2index[processed_token])
                else:
                    string_seq.append(self.token2index.get(processed_token, self.UNK.hash))

        if add_eos:
            string_seq.append(self.EOS.hash)

        return string_seq


    def indices2tokens(self, indices, ignore_ids=(), oovs=(),
                       highlight_oovs=False):

        tokens = []

        for idx in indices:
            if idx in ignore_ids:
                continue
            if idx >= len(self.index2token):
                if oovs:
                    oov_idx = idx - len(self.index2token) - 1
                    oov_str = oovs[oov_idx]
                    if highlight_oovs:
                        tokens.append('__{}__'.format(oov_str))
                    else:
                        tokens.append(oov_str)
                else:
                    raise IndexError(idx)
            else:
                tokens.append(self.index2token[idx].string)

        return tokens
