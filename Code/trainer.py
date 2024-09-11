import torch
import torch.nn as nn
from torch import optim
from encoder_lstm import Encoder
from decoder_lstm import Decoder
from lang import Lang
from tqdm import tqdm
from torcheval.metrics.functional.text.bleu import bleu_score
import copy


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

        target_tensor = target_tensor.view(-1)

        # decoder output and target must be of same length
        length_diff = target_tensor.size(0) - decoder_outputs.size(0)
        if length_diff > 0:
            # decoder output is shorter than target - pad first dimension with 0s
            decoder_outputs = nn.functional.pad(decoder_outputs, pad=(0, 0, 0, length_diff),
                                                mode='constant', value=0)
        elif length_diff < 0:
            # decoder output is longer than target - remove outputs after target length
            indices = torch.tensor([i for i in range(target_tensor.size(0))]).to("cuda")
            decoder_outputs = torch.index_select(decoder_outputs, 0, indices)


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
        best_score = 0
        best_epoch = 0
        for epoch in range(1, n_epochs + 1):
            trainer.decoder.train()
            trainer.encoder.train()
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
                best_epoch = epoch
                best_encoder_state = copy.deepcopy(self.encoder.state_dict())
                best_decoder_state = copy.deepcopy(self.decoder.state_dict())
            ave_epoch_loss = loss_total / num_training_pairs
            if ave_epoch_loss < best_loss:
                best_loss = ave_epoch_loss


            print(f"({epoch} {epoch/n_epochs*100:.0f}%) {ave_epoch_loss:.10f}")
        print(f"target: {best_target}\nbest_translation: {best_trans}\nbest_score: {best_score}\nbest_epoch: {best_epoch}")
        return {"best_encoder": best_encoder_state,
                "best_decoder": best_decoder_state,
                "best_epoch": best_epoch,
                "score": best_score
                }

def evaluate_sentence(source, target):

    with torch.no_grad():
        trainer.encoder.eval()
        trainer.decoder.eval()
        # use the encoder and decoder to get the predicted translation
        encoder_outputs, encoder_hidden = trainer.encoder(source)
        decoder_outputs, decoder_hidden = trainer.decoder(encoder_hidden)

        decoded_tensors = torch.argmax(decoder_outputs, dim=1)
        decoded_words = [eng_chi.idx2token[idx.item()] for idx in decoded_tensors]

        candidates = " ".join(decoded_words)
        references = " ".join(target)

        n_gram = 3
        score = bleu_score(candidates, [references], n_gram=n_gram)
        return decoded_words, score

def evaluate():

    scores_sum = 0
    for input_sent, target in zip(trainer.dev_in_tensors, trainer.dev_target_strings):
        input_sent = input_sent.to("cuda")
        translation, score = evaluate_sentence(input_sent, target)
        scores_sum += score
    print(f"input:\t{input_sent}\ntarget:\t{target}\ntrans:\t{translation}\nscore: {scores_sum / len(trainer.dev_target_strings)}")
    return target, translation, scores_sum / len(trainer.dev_target_strings)



lang = "chinese"
eng_chi = Lang()
in_amr, out_amr = eng_chi.load_data("../Data/cleaned_amr_data2.csv", "farsi")
in_amr_tokenized = eng_chi.tokenize(in_amr, True)
out_amr_tokenized = eng_chi.tokenize(out_amr, False)
eng_chi.set_mappings()

# 90% training and 10% testing
in_amr_tokenized = in_amr_tokenized[:-150]
out_amr_tokenized = out_amr_tokenized[:-150]
dev_in_amr_tokenized = in_amr_tokenized[-150:]
dev_out_amr_tokenized = out_amr_tokenized[-150:]
dev_in_amr_tokenized_tensors = eng_chi.create_tensors(dev_in_amr_tokenized)
eng_chi.in_amr_tensors = eng_chi.create_tensors(in_amr_tokenized)
eng_chi.out_amr_tensors = eng_chi.create_tensors(out_amr_tokenized)


# set hyperparameters
n_epochs = 100
hidden_size = 256
learning_rate = 0.00001
teacher_forcing_ratio = 1.0
bidirectional = True
n_layers = 2
dropout = 0.2

encoder = Encoder(len(eng_chi.vocab), n_layers, hidden_size, dropout=dropout)
decoder = Decoder(hidden_size, len(eng_chi.vocab), n_layers, eng_chi.token2idx["<EOS>"], dropout=dropout, teacher_forcing_ratio=teacher_forcing_ratio)

encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

trainer = Trainer(eng_chi.in_amr_tensors, eng_chi.out_amr_tensors, dev_in_amr_tokenized_tensors, dev_out_amr_tokenized,
                  encoder, decoder,
                  encoder_optimizer, decoder_optimizer,
                  criterion)

model_dict = trainer.train(n_epochs)
model_dict["epochs"] = n_epochs
model_dict["hidden_size"] = hidden_size
model_dict["lr"] = learning_rate
model_dict["tfr"] = teacher_forcing_ratio
model_dict["bidirectional"] = bidirectional
model_dict["n_layers"] = n_layers
model_dict["dropout"] = dropout

torch.save(model_dict, f"../models/best_model_trigram_{lang}.pth")