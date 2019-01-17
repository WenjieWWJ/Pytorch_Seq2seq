# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import torch


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkown token

MAX_POST_LENGTH = 100  # Maximum post length to consider
MAX_RESPONSE_LENGTH = 30  # Maximum response length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming



class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD" : PAD_token, "SOS" : SOS_token, "EOS" : EOS_token, "UNK" : UNK_token}
        self.word2count = {"PAD" : 1, "SOS" : 1, "EOS" : 1, "UNK" : 1}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD" : PAD_token, "SOS" : SOS_token, "EOS" : EOS_token, "UNK" : UNK_token}
        self.word2count = {"PAD" : 1, "SOS" : 1, "EOS" : 1, "UNK" : 1}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token : "UNK"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def sentence2index(self, sentence):
        indexes = []
        for word in sentence.split(' '):
            if word in self.word2index:
                indexes.append(self.word2index[word])
            else:
                indexes.append(self.word2index["UNK"])
        indexes.append(EOS_token) # add EOS at the end of every sentence
        return indexes

    def index2sentence(self, indexes):
        words = []
        for index in indexes:
            if index in self.index2word:
                words.append(self.index2word[word])
            else:
                words.append("OOV") # out of vocab, it doesn't usually happen
        return words


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[()&/]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Read query/response pairs and return a voc object
def loadData(data_dir, post_name, response_name):

    # Read the file and split into lines
    post_lines = open(os.path.join(data_dir, post_name), encoding='utf-8').read().strip().split('\n')
    response_lines = open(os.path.join(data_dir, response_name), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    if len(post_lines) != len(response_lines):
        print("data input error: len(post_lines) != len(response_lines)")
    pairs = []
    for p, r in zip(post_lines, response_lines):
        pairs.append([normalizeString(p), normalizeString(r)])

    return pairs

# modified by wenjie: clip the sentence when the length >= MAX_LENGTH

def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    if len(p[0].split(' ')) >= MAX_POST_LENGTH:
        p[0] = " ".join(p[0].split()[0:MAX_POST_LENGTH])
    if len(p[1].split(' ')) >= MAX_RESPONSE_LENGTH:
        p[1] = " ".join(p[1].split()[0:MAX_RESPONSE_LENGTH])
    return p


# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [filterPair(pair) for pair in pairs]

# Load training data and construct vocab
def loadTrainingData(data_dir, post_name, response_name, save_dir):

    print("loading Data ...")
    pairs = loadData(data_dir, post_name, response_name)

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)

    voc = Vocabulary("seq2seqVocab")
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)

    # save vocabulary to save_dir
    file = open(os.path.join(save_dir, "vocab.txt"), "w")
    vocab = [" ".join([str(index), word, str(voc.word2count[word])]) for index, word in voc.index2word.items()]
    file.write("\n".join(vocab))
    print("Save vocab done")
    
    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):

    indexes_batch = [voc.sentence2index(sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    
    indexes_batch = [voc.sentence2index(sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

