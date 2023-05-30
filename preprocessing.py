
from transformers import AutoTokenizer

import pandas as pd
import torch
from torch.utils.data import random_split
from utils.tokenize_vocab import *


df = pd.read_csv('/data/ezc2105/uniprot-transformer/uniprot.tsv',sep='\t')

# Drop proteins with no sequence or no molecular function label
df = df.dropna(subset=['Sequence'])
df = df.dropna(subset=['Gene Ontology (molecular function)'])

# Select molecular functions with at least 500 associated proteins
counts = df['Gene Ontology (molecular function)'].value_counts()
valid = counts[counts > 500].index
df = df[df['Gene Ontology (molecular function)'].isin(valid)]


# Select shorter proteins for gpu memory space
df = df[df['Sequence'].str.len() < 500]

# We choose these two labels:                                                    
# metal ion binding [GO:0046872]                                                      1785
# translation elongation factor activity [GO:0003746]                                 1760    
# transmembrane transporter activity [GO:0022857]                                     1291
# NADH dehydrogenase (ubiquinone) activity [GO:0008137]                               1164


df['Gene Ontology (molecular function)'].value_counts().head(19)
df = df[df['Gene Ontology (molecular function)'].isin(['metal ion binding [GO:0046872]','translation elongation factor activity [GO:0003746]','transmembrane transporter activity [GO:0022857]','NADH dehydrogenase (ubiquinone) activity [GO:0008137]'])]


# Integer encode labels
label_indices = df['Gene Ontology (molecular function)'].replace(['metal ion binding [GO:0046872]','translation elongation factor activity [GO:0003746]','transmembrane transporter activity [GO:0022857]','NADH dehydrogenase (ubiquinone) activity [GO:0008137]'], [0,1,2,3])
# Encoding scheme:
# metal ion binding [GO:0046872]                                                0
# translation elongation factor activity [GO:0003746]                           1    
# transmembrane transporter activity [GO:0022857]                               2
# NADH dehydrogenase (ubiquinone) activity [GO:0008137]                         3

# x and y
x = df['Sequence'].tolist()
y = label_indices.values

dataset = list(zip(x,y))

# train test split
train_set, val_set, test_set = random_split(dataset, [.8, .1, .1])


# info needed in run_training
num_labels = 4
max_seq_len = 0
for item in x:
    max_seq_len = max(len(item), max_seq_len)

print("number of labels: ", num_labels)
print("max seq len: ", max_seq_len)



# %%




# import pandas as pd
# import torch
# from torch.utils.data import random_split
# from utils.tokenize_vocab import *


# df = pd.read_csv('/data/ezc2105/uniprot-transformer/uniprot.tsv',sep='\t')

# # Drop proteins with no sequence or no molecular function label
# df = df.dropna(subset=['Sequence'])
# df = df.dropna(subset=['Gene Ontology (molecular function)'])

# # Select molecular functions with at least 1000 associated proteins
# counts = df['Gene Ontology (molecular function)'].value_counts()
# valid = counts[counts > 1000].index
# df = df[df['Gene Ontology (molecular function)'].isin(valid)]

# # Select shorter proteins for gpu memory space
# df = df[df['Sequence'].str.len() < 500]
# df[df['Gene Ontology (molecular function)'].isin(['DNA binding [GO:0003677]; DNA-binding transcription factor activity [GO:0003700]'])]


# # We choose these two labels:
# # DNA binding [GO:0003677]; DNA-binding transcription factor activity [GO:0003700]    2245 (2334 if we take larger proteins too)
# # transmembrane transporter activity [GO:0022857]                                     1291 (1948 if we take larger proteins too)
# # total: 4282 proteins
# df['Gene Ontology (molecular function)'].value_counts().head(15)
# df = df[df['Gene Ontology (molecular function)'].isin(['DNA binding [GO:0003677]; DNA-binding transcription factor activity [GO:0003700]','transmembrane transporter activity [GO:0022857]'])]


# # Select columns we care about
# raw_seq = df['Sequence']
# raw_label = df['Gene Ontology (molecular function)']


# # Tokenize labels and integer encode
# label_vocab = Vocab(raw_label.values.tolist())
# label_indices = raw_label.apply(label_vocab.to_index)

# # Tokenize sequence (split into list of 1 amino acids-long tokens) and integer encode
# seq_tokens = raw_seq.apply(tokenize_string, args=(1,))
# seq_vocab = Vocab(seq_tokens.values.tolist(), has_padding=True)
# seq_indices = seq_tokens.apply(seq_vocab.to_indices)
#     #seq_indices is a list of indices


# x = seq_indices.values
# y = label_indices.values

# # make x list of tensors
# x = [torch.tensor(item) for item in x]


# src_vocab_size = len(seq_vocab)
# num_labels = len(label_vocab)


# # save so we don't have to repeatedly reprocess
# torch.save(x, 'x.pt')
# torch.save(y, 'y.pt')
# torch.save(src_vocab_size, 'src_vocab_size.pt')
# torch.save(num_labels, 'num_labels.pt')
# """
# x = torch.load('x.pt')
# y = torch.load('y.pt')
# src_vocab_size = torch.load('src_vocab_size.pt')
# num_labels = torch.load('num_labels.pt')

# """
# # train test split
# x_train, x_val, x_test = random_split(x, [.8, .1, .1])
# y_train, y_val, y_test = random_split(y, [.8, .1, .1])

# # datasets
# train_set = list(zip(x_train,y_train))
# val_set = list(zip(x_val,y_val))
# test_set = list(zip(x_test,y_test))


# # info needed in run_training
# src_vocab_size = src_vocab_size
# num_labels = num_labels
# max_seq_len = 0
# for item in x:
#     max_seq_len = max(len(item), max_seq_len)

# print("number of sequence tokens / src vocab size: ", src_vocab_size)
# print("number of labels: ", num_labels)
# print("max seq len: ", max_seq_len)



# # %%
