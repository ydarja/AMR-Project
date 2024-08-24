import torch
import torch.nn as nn
from torch import optim
from encoder import Encoder
from decoder import Decoder
from lang import Lang
from tqdm import tqdm
from torcheval.metrics.functional.text.bleu import bleu_score

class Trainer():
    def __init__(self, in_tensor, out_tensor, dev_in_tensors, dev_target_strings,
                 encoder, decoder,
                 encoder_optimizer, decoder_optimizer,
                 criterion):

        self.in_tensor = in_tensor
        self.out_tensor = out_tensor
        self.dev_in_tensors = dev_in_tensors
        self.dev_target_strings = dev_target_strings
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion


    def _train_one(self, input_tensor, target_tensor):

        # zero optimizer gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # forward pass on encoder and decoder
        decoder_output = None

        encoder_outputs, encoder_hidden = self.encoder(input_tensor)

        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = self.decoder(decoder_hidden, target_tensor)

        # Create a view of target_tensor that removes the last dimension.
        # This is ok because the last dimension is always 1 (1 token index).
        # Example: target_tensor shape (5, 1) --> (5,)
        target_tensor = target_tensor.view(-1)

        # The number of decoder outputs must be the same as the number of tokens in target
        # in order to compute the loss.
        # Either chop off extra predictions, or pad with 0's
        length_diff = target_tensor.size(0) - decoder_outputs.size(0)
        if length_diff > 0:
            # decoder output is shorter than target - pad first dimension with 0s
            decoder_outputs = nn.functional.pad(decoder_outputs, pad=(0, 0, 0, length_diff),
                                                mode='constant', value=0)
        elif length_diff < 0:
            # decoder output is longer than target - remove outputs after target length
            indices = torch.tensor([i for i in range(target_tensor.size(0))]).to("cuda")
            decoder_outputs = torch.index_select(decoder_outputs, 0, indices)

        # Compute loss for this sample.
        # NLLLoss expects two tensors:
        #   predictions: tensor of shape (target_tensor.size(0), output vocab_size) containing
        #       a log probability distribution for each output
        #   gold values: target tensor of shape (num_tokens,)
        loss = self.criterion(decoder_outputs, target_tensor)

        # backpropagation
        loss.backward()

        nn.utils.clip_grad_norm(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm(decoder.parameters(), 1.0)

        # update weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # return decoder output and scalar loss
        return decoder_output, loss.item()

    def train(self, n_epochs):

        if torch.cuda.is_available():
            self.encoder = self.encoder.to(torch.device("cuda"))
            self.decoder = self.decoder.to(torch.device("cuda"))
        print("training...")

        # number of training pairs
        num_training_pairs = len(self.in_tensor)

        best_loss = float("inf")
        loss = float("inf")
        best_score = 0
        for epoch in range(1, n_epochs + 1):
            loss_total = 0
            for pair_idx in tqdm(range(num_training_pairs)):

                input_tensor = self.in_tensor[pair_idx].to("cuda")
                target_tensor = self.out_tensor[pair_idx].to("cuda")

                decoder_output, loss = self._train_one(input_tensor, target_tensor)

                loss_total += loss
            target, translation, score = evaluate()
            if score > best_score:
                best_target = target
                best_score = score
                best_trans = translation
            ave_epoch_loss = loss_total / num_training_pairs
            if ave_epoch_loss < best_loss:
                best_loss = ave_epoch_loss

            print(f"({epoch} {epoch/n_epochs*100:.0f}%) {ave_epoch_loss:.10f}")
        print(f"target: {best_target}\nbest_translation: {best_trans}\nbest_score: {best_score}")

def evaluate_sentence(source, target):
    """
    Evaluate and return the model prediction of one sentence
    and the bleu score (use the imported bleu_score function) calculated
    against target, with n-gram=1.
    Helper for evaluate().

    Note that bleu_score() requires inputs as strings, so you
    need to join the tokens.

    :param sentence: sentence as a list[str] of tokens
    :param target: target translation as a list[str] of tokens
    :return: model prediction as a list[str], and the bleu score
    """

    with torch.no_grad():

        # use the encoder and decoder to get the predicted translation
        encoder_outputs, encoder_hidden = trainer.encoder(source)
        decoder_outputs, decoder_hidden = trainer.decoder(encoder_hidden)

        decoded_tensors = torch.argmax(decoder_outputs, dim=1)
        decoded_words = [eng_chi.idx2token[idx.item()] for idx in decoded_tensors]

        candidates = " ".join(decoded_words)
        references = " ".join(target)

        n_gram = 1
        score = bleu_score(candidates, [references], n_gram=n_gram)
        return decoded_words, score

def evaluate(verbose=False):
    """
    Calculate the bleu score for each pair in self.io_token_pairs.
    Return the average bleu score over all pairs.

    :param verbose: if True, print input, target, translation, score for each sentence
    :return: average bleu score over all pairs in self.io_token_pairs
    """

    scores_sum = 0
    for input_sent, target in zip(trainer.dev_in_tensors, trainer.dev_target_strings):
        input_sent = input_sent.to("cuda")
        translation, score = evaluate_sentence(input_sent, target)
        scores_sum += score
        # if verbose:
    print(f"input:\t{input_sent}\ntarget:\t{target}\ntrans:\t{translation}\nscore: {scores_sum / len(trainer.dev_target_strings)}")
    return target, translation, scores_sum / len(trainer.dev_target_strings)




eng_chi = Lang()
in_amr, out_amr = eng_chi.load_data("../Data/cleaned_amr_data2.csv", "chinese")
in_amr_tokenized = eng_chi.tokenize(in_amr, True)
out_amr_tokenized = eng_chi.tokenize(out_amr, False)
eng_chi.set_mappings()
in_amr_tokenized = in_amr_tokenized[:-150]
out_amr_tokenized = out_amr_tokenized[:-150]
dev_in_amr_tokenized = in_amr_tokenized[-150:]
dev_out_amr_tokenized = out_amr_tokenized[-150:]
dev_in_amr_tokenized_tensors = eng_chi.create_tensors(dev_in_amr_tokenized)
eng_chi.in_amr_tensors = eng_chi.create_tensors(in_amr_tokenized)
eng_chi.out_amr_tensors = eng_chi.create_tensors(out_amr_tokenized)


# set hyperparameters
n_epochs = 30  # baseline 20, best 60
hidden_size = 32  # baseline 8, best 32
learning_rate = 0.001  # baseline .01, best .001
teacher_forcing_ratio = 0.5  # baseline 0, don't use tf, best .5
bidirectional = True  # baseline False, uni-directional, best True

encoder = Encoder(len(eng_chi.vocab), hidden_size)
decoder = Decoder(hidden_size, len(eng_chi.vocab), teacher_forcing_ratio=teacher_forcing_ratio)

encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

trainer = Trainer(eng_chi.in_amr_tensors, eng_chi.out_amr_tensors, dev_in_amr_tokenized_tensors, dev_out_amr_tokenized,
                  encoder, decoder,
                  encoder_optimizer, decoder_optimizer,
                  criterion)

trainer.train(n_epochs)
