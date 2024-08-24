import csv
import torch

class Lang:

    def __init__(self):
        self.vocab = []
        self.token2idx = {} # map token to index
        self.idx2token = {} # map index to token
        self.in_amr_tensors = None # list of tensors of indices
        self.out_amr_tensors = None # list of tensors of indices

    def load_data(self, path, language):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            # skip header
            next(reader)
            in_amrs = []
            out_amrs = []
            for row in reader:
                # select english AMR
                in_amrs.append(row[2])
                # select chinese or farsi AMR
                out_amr = row[4] if language == "chinese" else row[6]
                if out_amr == None or out_amr == "" or row[2] == "" or row[2] == None:
                    print("alert")
                out_amrs.append(out_amr)

        return in_amrs, out_amrs

    def tokenize(self, tlp_amrs, source):

        self.vocab.append("<BOS>")
        self.vocab.append("<EOS>")
        res = []
        for sent in tlp_amrs:
            sent = sent.split()
            for t in sent:
                if t not in self.vocab:
                    self.vocab.append(t)
            if source:
                res.append(["<BOS>"] + sent + ["<EOS>"])
            else:
                res.append(sent + ["<EOS>"])
        return res

    def set_mappings(self):
        self.token2idx = {token: i for i, token in enumerate(self.vocab)}
        self.idx2token = {v: k for k, v in self.token2idx.items()}


    def create_tensors(self, amr):
        res = []
        for sent in amr:
            sent_t = []
            for token in sent:
                sent_t.append(self.token2idx[token])
            # transpose row to column vector
            res.append(torch.tensor(sent_t, dtype=torch.long).reshape(-1, 1))
        return res