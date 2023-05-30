import pandas as pd
from collections import Counter


"""
split string into list where each item is n characters long
"""
def tokenize_string(string, n=1):
    out = [(string[i:i+n]) for i in range(0, len(string), n)]
    return out

"""
Takes in 1D or 2D list of tokens
    token_freqs:
"""
class Vocab:
    def __init__(self, tokens=[], has_padding=False):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            print("flattened 2d list when generating vocab")
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # token_freqs is dictionary of (unique_tokens:freq) in descending order of freq
        # The list of unique tokens   
        if has_padding==False:
            self.idx_to_token = list(token for token, freq in self.token_freqs)
        else:
            # save 0 for pad if pad=True
            self.idx_to_token = list()
            self.idx_to_token.append("pad")
            for token, freq in self.token_freqs:
                self.idx_to_token.append(token)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    """
    input: list of tokens or indices
    output: list of indices or tokens
    """
    def to_indices(self, tokens):
        return [self.token_to_idx[token] for token in tokens]

    def to_tokens(self, indices):
        return [self.idx_to_token[int(index)] for index in indices]

    """
    input: single token or index
    output: single index or token
    """
    def to_index(self, token):
        return self.token_to_idx[token]
    
    def to_token(self, index):
        return self.idx_to_token[int(index)]