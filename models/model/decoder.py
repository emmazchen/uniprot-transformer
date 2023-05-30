from torch import nn

from models.blocks.embedding_block import EmbeddingBlock
from models.blocks.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, n_blocks, trg_vocab_size, max_seq_len, d_embedding, ff_d_hidden, n_heads, p_drop):
        super().init()
        self.n_blocks=n_blocks
        self.emb = EmbeddingBlock(d_embedding=d_embedding, vocab_size=trg_vocab_size, max_seq_len=max_seq_len, p_drop=p_drop)
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_embedding=d_embedding, ff_d_hidden=ff_d_hidden, n_heads=n_heads, p_drop=p_drop) 
                                         for i in range(n_blocks)])


    def forward(self, trg, enc_outputs, src_mask, trg_mask):
        x = self.emb(x) # embed

        for block in self.dec_blocks: # pass through all the dec blocks
            x = block(dec=trg, enc=enc_outputs, src_mask=src_mask, trg_mask=trg_mask)
        
        return x
