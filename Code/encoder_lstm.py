import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, n_layer, hidden_size, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.n_layer = n_layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layer, batch_first=True, bidirectional=True, dropout=dropout)

    def forward_step(self, x, hidden):

        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def forward(self, x):

        # initialize hidden state
        hidden = torch.zeros(2*self.n_layer, self.hidden_size).to("cuda")
        cell = torch.zeros(2*self.n_layer, self.hidden_size).to("cuda")
        outputs = []

        # iterate token tensors in x
        input_length = x.size(0)
        for ei in range(input_length):
            output, (hidden, cell) = self.forward_step(x[ei], (hidden, cell))
            outputs.append(output)

        hidden = (hidden[:self.n_layer, :] + hidden[self.n_layer:, :])
        cell = (cell[:self.n_layer, :] + cell[self.n_layer:, :])

        # return values
        return outputs, (hidden, cell)