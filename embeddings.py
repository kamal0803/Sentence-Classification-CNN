import os
import torch
from gensim.models import KeyedVectors
import numpy as np

def load_word2vec():
    word2vec_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "word2vec-google-news-300",
        "word2vec-google-news-300.gz",
    )
    return KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def build_vocab(tokenised_sentences):
    vocab = {}
    vocab['PAD'] = 0
    vocab['UNK'] = 1
    for sentence in tokenised_sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

def build_embedding_matrix_random(vocab, dim=300):
    emb_matrix = torch.zeros(len(vocab), dim)

    for word, index in vocab.items():
        if word == 'PAD':
            emb_matrix[index] = torch.zeros(300)
        else:
            emb_matrix[index] = torch.randn(300) * 0.01

    return emb_matrix

def build_embedding_matrix(vocab, wv, dim=300):
    emb_matrix = torch.zeros(len(vocab), dim)

    for word, index in vocab.items():
        if word == "PAD":
            emb_matrix[index] = torch.zeros(dim)
        elif word == "UNK":
            emb_matrix[index] = torch.randn(dim) * 0.01
        elif word in wv:
            emb_matrix[index] = torch.tensor(wv[word], dtype=torch.float32)
        else:
            emb_matrix[index] = torch.randn(dim) * 0.01

    return emb_matrix


