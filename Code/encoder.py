import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    The encoder of a seq2seq network is an RNN that takes each token in a
    sequence and the final hidden tensor is an encoding of the entire input.

    The final hidden tensor is used as the initial hidden tensor of the decoder.
    """
    def __init__(self, input_vocab_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward_step(self, x, hidden):
        """
        One step of a forward pass.
        Returns:
        1: the output (1, hidden_size * directions)
        2: the new hidden tensor of shape (directions, hidden_size)

        where directions = 2 if bidirectional, else 1

        :param x: token tensor of shape (1,)
        :param hidden: hidden tensor of shape (directions, hidden_size)
        :return: output of shape (1, hidden_size * directions), hidden tensor of shape (directions, hidden_size)
        """
        embedded = self.embedding(x)  # (1,hidden_size)
        output, hidden = self.gru(embedded, hidden)  # (1, hidden_size * directions), (directions, hidden_size)
        return output, hidden

    def forward(self, x):

        # initialize hidden state
        hidden = torch.zeros(2, self.hidden_size).to("cuda")
        outputs = []

        # iterate token tensors in x
        input_length = x.size(0)
        for ei in range(input_length):
            output, hidden = self.forward_step(x[ei], hidden)
            outputs.append(output)

        hidden = (hidden[0, :] + hidden[1, :]).unsqueeze(dim=0)

        # return values
        return outputs, hidden