from torch.nn.utils.rnn import pad_sequence
import torch

def padding_collate(data):
    (x, y) = zip(*data)

    x = torch.tensor(pad_sequence(list(x), batch_first=True))
    y = torch.tensor(y)

    return x, y