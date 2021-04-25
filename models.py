import torch

# defining model definition
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, device):
        super(Model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)
        self.output = torch.nn.Linear(hidden_dim, vocab_size).to(device=device)

    def forward(self, data, hidden):
        embedded = self.embedding(data)
        prediction, hidden = self.rnn(embedded, hidden)
        return self.output(prediction), hidden

    def init_hidden(self, BATCH_SIZE):
        return torch.zeros(1, BATCH_SIZE, self.hidden_dim), torch.zeros(1, BATCH_SIZE, self.hidden_dim, device=self.device)