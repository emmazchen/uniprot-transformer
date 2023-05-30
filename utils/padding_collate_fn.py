from torch.nn.utils.rnn import pad_sequence
import torch

def padding_collate(data):
    (x, y) = zip(*data)

    x = torch.tensor(pad_sequence(list(x), batch_first=True))
    y = torch.tensor(y)

    return x, y


from transformers import EsmTokenizer

def esm_collate(data):
    (x, y) = zip(*data)
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    out = tokenizer(x, padding=True, return_tensors="pt")
    x = out['input_ids']
    mask = out['attention_mask']
    
    y = torch.tensor(y)

    return x, y, mask