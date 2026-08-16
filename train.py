import torch

from data import get_word_indices


class EarlyStopping:

    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = -float("inf")
        self.epochs_without_improvement = 0

    def step(self, score):
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.should_stop

    @property
    def should_stop(self):
        return self.epochs_without_improvement >= self.patience


def train_model(model, train_sentences, Y_train, test_sentences, Y_test, vocab, epochs=10, patience=3, min_delta=0.01):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(epochs):

        total_loss = 0.0
        correct_train = 0
        correct_test = 0

        model.train()
        for i in range(len(train_sentences)):

            X = get_word_indices(train_sentences[i], vocab)
            y = Y_train[i]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            prediction = torch.argmax(logits)

            if prediction == y:
                correct_train += 1

        avg_loss = total_loss / len(train_sentences)
        train_accuracy = correct_train / len(train_sentences)

        print(f"Epoch {epoch + 1}, Train Loss = {avg_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")

        model.eval()
        with torch.no_grad():

            for j in range(len(test_sentences)):
                X_eval = get_word_indices(test_sentences[j], vocab)
                y_test = Y_test[j]
                logits = model(X_eval)
                prediction = torch.argmax(logits)

                if prediction == y_test:
                    correct_test += 1

        test_accuracy = correct_test / len(test_sentences)

        print(f"Epoch {epoch + 1}, Test Accuracy = {test_accuracy:.4f}")

        if early_stopping.step(test_accuracy):
            print(f"Early stopping at epoch {epoch + 1} (best Test Accuracy = {early_stopping.best_score:.4f})")
            break

    return model, early_stopping.best_score
