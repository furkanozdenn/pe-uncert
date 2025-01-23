import numpy as np
import pdb

"""
Generates tokens for 3-mer model
Single sequence tokens (sequences are not merged)
"""

k_mer = 3

# chars are 'A', 'C', 'G', 'T', 'N' for off-target sequences
char_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4} # TODO: add '-' for bulges

# generate all possible tokens
tokens = []
for char1 in char_dict:
    for char2 in char_dict:
        for char3 in char_dict:
            tokens.append(char1 + char2 + char3)

# generate token to index dictionary
token_to_index = {}
for i, token in enumerate(tokens):
    token_to_index[token] = i

# generate index to token dictionary
index_to_token = {}
for i, token in enumerate(tokens):
    index_to_token[i] = token

# write token to index dictionary to file
filename = f'single_seq_{k_mer}mer_tokens.txt'
with open(filename, 'w') as f:
    for token in token_to_index:
        f.write(token + '\t' + str(token_to_index[token]) + '\n')