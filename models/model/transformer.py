import torch
import torch.nn as nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = eval(mconfigs['pad_idx'])
        self.encoder = Encoder(**mconfigs['encoder'])
        self.decoder = Decoder(**mconfigs['decoder'])
        self.linear = nn.Linear(**mconfigs['decoder']['d_embedding'], **mconfigs['decoder']['trg_vocab_size'])

    def forward(self, src, trg):
        """
        Inputs size:
            src, trg: (batch_size, src_seq_len)
        Outputs size:
            (batchsize, seq_len, trg_vocab_size) 
            -> logits (we don't apply softmax because handled by loss function)
        """

        # make masks
        src_mask = self.make_pad_mask(src, src, self.pad_idx) # used in self attentn in encoder
        trg_src_mask = self.make_pad_mask(trg, src, self.pad_idx) # used in encoder-decoder attn in decoder
        trg_mask = self.make_pad_mask(trg, trg, self.pad_idx) * self.make_look_ahead_mask(trg, trg) # used in self attentn in decoder
        
        # pass through blocks
        enc_outputs = self.encoder(x=src, src_mask=src_mask)
        dec_outputs = self.decoder(dec=trg, enc_outputs=enc_outputs, src_mask=src_mask, trg_mask=trg_mask)
        out = self.linear(dec_outputs) #don't perform softmax because handled by loss function
        return out



    """
    Helper methods to make masks
    """

    def make_pad_mask(self, q, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            we apply mask on each of the heads, which is why it has size 1 along dim 1
        """

        len_q, len_k = q.size(1), k.size(1) # max_seq_len for q and k, i.e. the len to which all seq are padded to

        q_2d = q.ne(pad_idx)
        k_2d = k.ne(pad_idx) # these are 2d: (batch_size, src_seq_len)

        # make 4d, then expand along dims to reach final size (batch_size, 1, len_q, len_k)
        k = k_2d.unsqueeze(1).unsqueeze(2) 
        q = q_2d.unsqueeze(1).unsqueeze(3) # k is now (batch_size, 1, 1, len_k), q is now (batch_size, 1, len_q, 1)
        k = k.repeat(1, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # k and q now (batch_size, 1, len_q, len_k)

        # effect of matmul -> only valid when both are not padding
        mask = k & q 
        return mask



    def make_look_ahead_mask(self, q, k):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
        """

        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1) # get seq len of q and k

        # 2d lower triangle matrix - everything above main diag is 0
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)

        # make 4d and size (batch_size, 1, seq_len_q, seq_len_k)
        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.repeat(batch_size, 1, 1, 1)

        return mask


 