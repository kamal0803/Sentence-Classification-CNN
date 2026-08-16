import nltk
import torch
from data import prepare_trec_data, prepare_sst2_data, preprocess
from embeddings import build_embedding_matrix, build_embedding_matrix_random, build_vocab, load_word2vec
from model import SentenceClassifierCNNMultiChannel, SentenceClassifierCNNSingleChannel
from train import train_model

nltk.download('punkt_tab')
nltk.download('stopwords')

def main():
    # train_sentences, Y_train = prepare_trec_data("train")
    # train_new_sentences = preprocess(train_sentences)
    # test_sentences, Y_test = prepare_trec_data("test")
    # test_new_sentences = preprocess(test_sentences)

    train_sentences, Y_train = prepare_sst2_data("train")
    train_new_sentences = preprocess(train_sentences)
    test_sentences, Y_test = prepare_sst2_data("test")
    test_new_sentences = preprocess(test_sentences)
    
    out_features = len(torch.unique(Y_test))

    wv = load_word2vec()

    vocab = build_vocab(train_new_sentences)

    emb_matrix_random = build_embedding_matrix_random(vocab)
    emb_matrix = build_embedding_matrix(vocab, wv)

    embedding_random = torch.nn.Embedding.from_pretrained(emb_matrix_random, freeze=False, padding_idx=vocab["PAD"])
    embedding_static = torch.nn.Embedding.from_pretrained(emb_matrix, freeze=True, padding_idx=vocab["PAD"])
    embedding_non_static = torch.nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=vocab["PAD"])

    # A fresh embedding table is built for model 4
    embedding_static_mc = torch.nn.Embedding.from_pretrained(emb_matrix, freeze=True, padding_idx=vocab["PAD"])
    embedding_non_static_mc = torch.nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=vocab["PAD"])

    model_random = SentenceClassifierCNNSingleChannel(embedding_random, out_features=out_features)
    model_static = SentenceClassifierCNNSingleChannel(embedding_static, out_features=out_features)
    model_non_static = SentenceClassifierCNNSingleChannel(embedding_non_static, out_features=out_features)
    model_multi_channel = SentenceClassifierCNNMultiChannel(embedding_static_mc, embedding_non_static_mc, out_features=out_features)

    print("================RUNNING MODEL 1 - RANDOM==================================")
    _, best_accuracy_random = train_model(model_random, train_new_sentences, Y_train, test_new_sentences, Y_test, vocab, epochs=10)

    print("================RUNNING MODEL 2 - STATIC (static word2vec)==================================")
    _, best_accuracy_static = train_model(model_static, train_new_sentences, Y_train, test_new_sentences, Y_test, vocab, epochs=10)

    print("================RUNNING MODEL 3 - NON STATIC (fine tuned word2vec)==================================")
    _, best_accuracy_non_static = train_model(model_non_static, train_new_sentences, Y_train, test_new_sentences, Y_test, vocab, epochs=10)

    print("================RUNNING MODEL 4 - MULTI CHANNEL==================================")
    _, best_accuracy_multi_channel = train_model(model_multi_channel, train_new_sentences, Y_train, test_new_sentences, Y_test, vocab, epochs=10)

    print("================BEST TEST ACCURACY PER MODEL==================================")
    print(f"Random: {best_accuracy_random:.4f}")
    print(f"Static: {best_accuracy_static:.4f}")
    print(f"Non-static: {best_accuracy_non_static:.4f}")
    print(f"Multi-channel: {best_accuracy_multi_channel:.4f}")

if __name__ == "__main__":
    main()
