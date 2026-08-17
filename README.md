# Sentence Classification CNN

Reproduced the paper *Convolutional Neural Networks for Sentence Classification*, Kim (2014) in Pytorch. It trains a CNN over pretrained word2vec embeddings to classify sentences, and implements all four model variants described in the paper.

## Model variants

The CNN architecture has parallel filters of width 3, 4, and 5 over the embedding dimension, followed by max-over-time pooling, concatenation, dropout, and a final linear layer. The same hyperparameters are used as in the original paper, with 100 filters and Dropout of 0.5.

The 4 model variants are:

* **rand** — word embeddings are initialized randomly and learned from scratch.
* **static** — word embeddings are initialized from word2vec and kept frozen during training.
* **non-static** — word embeddings are initialized from word2vec and fine-tuned during training.
* **multichannel** — two embedding channels (one static, one non-static) are convolved separately and the resulting feature maps are summed.

## Datasets

Although the original paper used 7 different datasets for experimentation, I used only 2 - TREC and SST-2. Dataset is added in a separate folder.

* **TREC** — 6-class question classification (`ABBR`, `DESC`, `ENTY`, `HUM`, `LOC`, `NUM`), loaded from `trec_data/`.
* **SST-2** — binary sentiment classification, parsed from SST-5 parse trees in `sst2_data/` (neutral examples dropped, labels collapsed to positive/negative).

`main.py` currently trains on SST-2 and TREC is commented out.

## Preprocessing

Although I wasn't sure if preprocessing of text was performed in the original paper, I did some cleaning of text data by lower casing, removing hidden characters and stripping white spaces.

## Project structure

```
data.py         # dataset loading (TREC, SST-2) and text preprocessing/tokenisation
embeddings.py   # word2vec loading, vocab building, embedding matrix construction
model.py        # SentenceClassifierCNNSingleChannel, SentenceClassifierCNNMultiChannel
train.py        # training/evaluation loop, EarlyStopping
main.py         # entry point: builds data, embeddings, and all four model variants
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the pretrained [word2vec Google News vectors](https://code.google.com/archive/p/word2vec/) and place `word2vec-google-news-300.gz` in `word2vec-google-news-300/` at the project root.

## Running

```bash
python main.py
```

This trains all four model variants (rand, static, non-static, multichannel) sequentially with early stopping (patience=3, monitoring test accuracy), printing per-epoch loss/accuracy and a final best-accuracy.

## Results

Below are the results for best test accuracy per model variant, with early stopping applied.

Dataset | rand | static | non-static | multichannel
--- | --- | --- | --- | ---
TREC | 88.6 | **89.8** | 88 | 89.6
SST-2 | 73.64 | 77.81 | **80.18** | 76.6

## Reference

    Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014
    Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746-1751, Doha, Qatar.
    Association for Computational Linguistics.
