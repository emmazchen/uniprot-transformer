import pandas as pd
import torch
from torch.utils.data import random_split
from utils.tokenize_vocab import *
import json
from math import sqrt

#data
df = pd.read_csv('/data/ezc2105/uniprot-transformer/uniprot.tsv',sep='\t')

#config
configfile =  f"/data/ezc2105/dynamic-batch/configs/config.json"   # not sure if this works
with open(configfile) as m_stream:
        config = json.load(m_stream)


# Drop proteins with no sequence or no molecular function label
df = df.dropna(subset=['Sequence'])
df = df.dropna(subset=['Gene Ontology (molecular function)'])

# Select molecular functions with at least 1000 associated proteins
counts = df['Gene Ontology (molecular function)'].value_counts()
valid = counts[counts > 1000].index
df = df[df['Gene Ontology (molecular function)'].isin(valid)]


# Select shorter proteins for gpu memory space
max_token_num = config['max_num_token_per_batch'] # largest possible without cuda out of memory error - too big: 1e8
df = df[df['Sequence'].str.len() < sqrt(max_token_num)]


# Select columns we care about
raw_seq = df['Sequence']
raw_label = df['Gene Ontology (molecular function)']


# Tokenize labels and integer encode
label_vocab = Vocab(raw_label.values.tolist())
label_indices = raw_label.apply(label_vocab.to_index)

# Tokenize sequence (split into list of 1 amino acids-long tokens) and integer encode
seq_tokens = raw_seq.apply(tokenize_string, args=(1,))
seq_vocab = Vocab(seq_tokens.values.tolist(), has_padding=True)
seq_indices = seq_tokens.apply(seq_vocab.to_indices)


x = seq_indices.values
y = label_indices.values

x = [torch.tensor(item) for item in x]

dataset = list(zip(x,y))

# train test split
train_set, val_set, test_set = random_split(dataset, [.8, .1, .1])

# info needed in run_training
src_vocab_size = len(seq_vocab)
num_labels = len(label_vocab)
max_seq_len = 0
for item in x:
    max_seq_len = max(len(item), max_seq_len)

print("number of sequence tokens / src vocab size: ", len(seq_vocab))
print("number of labels: ", len(label_vocab))
print("max seq len: ", max_seq_len)





