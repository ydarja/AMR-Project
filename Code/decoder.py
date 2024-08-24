import torch
import torch.nn as nn
import random


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_vocab_size, teacher_forcing_ratio=0):
        super(Decoder, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_step(self, x, hidden):

        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.lin(output)
        output = self.log_softmax(output)
        return output, hidden

    def forward(self, encoder_hidden, target_tensor=None):

        # initialize decoder input and hidden tensors
        decoder_input = torch.tensor([0]).to("cuda")
        decoder_outputs = []
        decoder_hidden = encoder_hidden

        # if self.teacher_forcing_ratio > 0 and target_tensor is provided,
        # use teacher forcing for ~ teacher_forcing_ratio % of the inputs
        use_teacher_forcing = False
        if (target_tensor is not None) and (random.random() < self.teacher_forcing_ratio):
            use_teacher_forcing = True

        if use_teacher_forcing:
            target_length = target_tensor.size(0)
            for i in range(target_length):
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs.append(decoder_output)
                decoder_input = target_tensor[i]  # Teacher forcing

        else:
            for i in range(375):
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs.append(decoder_output)

                # Use own output as next input
                next_input = torch.argmax(decoder_output, dim=1)
                decoder_input = next_input

                if decoder_input.item() == 1:
                    break

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        # return values
        return decoder_outputs, decoder_hidden