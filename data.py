import torch
from spacy.lang.en import English
import numpy as np
from nltk.tree import Tree

def lowercase(sentences):

    for sentence in sentences:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].lower()

    return sentences

def remove_hidden_characters(sentences):

    for sentence in sentences:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].replace("\n", "")
            sentence[i] = sentence[i].replace("\t", "")
            sentence[i] = sentence[i].replace("\'", "")

    return sentences

def tokenise(sentences):

    nlp = English()
    modified_sentences = []

    for sentence in sentences:
        doc = nlp(sentence)
        data = []
        for token in doc:
            data.append(token.text)
        modified_sentences.append(data)

    return modified_sentences


def whitespace_remove(sentences):
    for sentence in sentences:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].strip()

    return sentences

def preprocess(sentences):

    data = tokenise(sentences)
    data = lowercase(data)
    data = remove_hidden_characters(data)
    data = whitespace_remove(data)

    return data

def get_word_indices(sentence, vocab):

    word_indices = []

    for i in range(len(sentence)):
        word = sentence[i]
        word_indices.append(vocab.get(word, vocab['UNK']))

    k = len(sentence)
    while k < 5:
        word_indices.append(0)
        k = k + 1

    word_indices = torch.tensor(word_indices, dtype=torch.long)

    return word_indices

def prepare_trec_data(split):

    label_map = {
        "ABBR": 0,
        "DESC": 1,
        "ENTY": 2,
        "HUM": 3,
        "LOC": 4,
        "NUM": 5
    }

    if split == "train":
        file_name = "./trec_data/trec_train.txt"

    elif split == "test":
        file_name = "./trec_data/trec_test.txt"


    sentences = []
    labels = []

    with open(file_name, "r", encoding="latin-1") as f:
        lines = f.readlines()

    for line in lines:

        label_info, sentence = line.strip().split(" ", 1)
        coarse_label = label_info.split(":")[0]
        sentences.append(sentence)
        labels.append(label_map[coarse_label])

    labels = torch.tensor(labels, dtype=torch.long)

    return sentences, labels

def prepare_sst2_data(data):
    
    if data == 'train':
        file = './sst2_data/train.txt'

    elif data == 'test':
        file = './sst2_data/test.txt'

    sentences = []
    labels = []
    
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
    
            if not line:
                continue
    
            tree = Tree.fromstring(line)
    
            # Root sentiment label: 0, 1, 2, 3, or 4
            sentiment = int(tree.label())
    
            # SST-2 removes neutral examples
            if sentiment == 2:
                continue
    
            # Reconstruct sentence from leaf nodes
            sentence = " ".join(tree.leaves())
    
            # Convert SST-5 labels to SST-2:
            # 0, 1 -> negative
            # 3, 4 -> positive
            label = 0 if sentiment < 2 else 1
    
            sentences.append(sentence)
            labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)

    return sentences, labels