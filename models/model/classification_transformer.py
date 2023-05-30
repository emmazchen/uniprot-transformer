import torch
import torch.nn as nn
from models.model.encoder import Encoder
from transformers import EsmModel


class ESMControl(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx']
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.avg_pool_reduce_dim = torch.mean # average along dimension to get (batch_size, d_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(mconfigs['encoder']['d_embedding'],mconfigs['mlp']['l1']['out_features']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3'])
        )

    def forward(self, x, mask):
        """
        Input size:
            x: (batch_size, seq_len)
        Output size:
            out: (batch_size, num_classes)
        """
        # make mask for self attention in encoder
        #src_mask = self.make_pad_mask(x, x, self.pad_idx)
        x = self.esm(x, mask).last_hidden_state
        # x size: batchsize * seqlen * dmodel
        x = self.avg_pool_reduce_dim(x, 1)
        out = self.mlp(x)
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
    
 
#%%


class ClassificationTransformer(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx']
        self.encoder = Encoder(**mconfigs['encoder'])
        self.avg_pool_reduce_dim = torch.mean # average along dimension to get (batch_size, d_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(mconfigs['encoder']['d_embedding'],mconfigs['mlp']['l1']['out_features']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3'])
        )
    
    def forward(self, x):
        """
        Input size:
            x: (batch_size, seq_len)
        Output size:
            out: (batch_size, num_classes)
        """
        # make mask for self attention in encoder
        src_mask = self.make_pad_mask(x, x, self.pad_idx)
 
        x = self.encoder(x, src_mask)
        x = self.avg_pool_reduce_dim(x, 1)
        out = self.mlp(x)
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

#%%
"""
Positive control with torch transformer
"""

from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from models.blocks.embedding_block import EmbeddingBlock

class TorchTransformerControl(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx']
        self.emb = EmbeddingBlock(mconfigs['encoder']['d_embedding'], mconfigs['encoder']['src_vocab_size'], mconfigs['encoder']['max_seq_len'], mconfigs['encoder']['p_drop'])
        encoder_layer = TransformerEncoderLayer(d_model=mconfigs['encoder']['d_embedding'], nhead=mconfigs['encoder']['n_heads'])
        self.encoder = TransformerEncoder(encoder_layer, mconfigs['encoder']['n_blocks'])
        self.avg_pool_reduce_dim = torch.mean # average along dimension to get (batch_size, d_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(mconfigs['encoder']['d_embedding'],mconfigs['mlp']['l1']['out_features']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3'])
        )
    
    def forward(self, x):
        """
        Input size:
            x: (batch_size, seq_len)
        Output size:
            out: (batch_size, num_classes)
        """
        # make mask for self attention in encoder
        src_mask = self.make_pad_mask_for_torch(x, self.pad_idx) #src mask should be (batchsize, seqlen)
        x = self.emb(x)
        x = torch.transpose(x, 0, 1) # convert from batch first
        x = self.encoder(x.float(), src_key_padding_mask=src_mask)
        x = torch.transpose(x, 0, 1) # convert back to batch first
        x = self.avg_pool_reduce_dim(x, 1)
        out = self.mlp(x)
        return out


    """
    Helper methods to make masks
    """
    

    def make_pad_mask_for_torch(self, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            k: (batch_size, seq_len)
        Output size:
            mask: (batch_size,  seq_len)
        """
        mask = k.ne(pad_idx)  # 0s where it's padding
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



#%%

