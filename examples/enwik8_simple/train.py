#!/usr/bin/env python3
import pdb

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 4096

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# instantiate model

model = ReformerLM(
    dim=512,
    depth=6,
    max_seq_len=SEQ_LEN,
    num_tokens=256,
    heads=8,
    bucket_size=64,
    n_hashes=4,
    ff_chunks=10,
    lsh_dropout=0.1,
    weight_tie=True,
    causal=True,
    n_local_attn_heads=4,
    use_full_attn=False  # set this to true for comparison with full attention
)

model = TrainingWrapper(model)
model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0,
                                   self.data.size(0) - self.seq_len - 1, (1, ))
        full_seq = self.data[rand_start:rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))

for i, x in enumerate(train_loader):
    print(x.shape)
    model.train()
    print(model)
    break
    loss = model(x, return_loss=True)

    if i == 4:
        break
