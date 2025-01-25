import numpy as np
import pdb
from itertools import product

def generate_vocab(k_mer=3):
    """
    Generates tokens and their corresponding indices for a k-mer model.
    
    Args:
        k_mer (int): The length of the k-mer. Defaults to 3.
    
    Returns:
        tuple: A tuple containing two dictionaries:
               - token_to_index: Mapping from token to its index.
               - index_to_token: Mapping from index to its token.
    """
    # Define characters for off-target sequences
    char_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}  # TODO: add '-' for bulges
    
    # Generate all possible tokens using Cartesian product
    tokens = [''.join(p) for p in product(char_dict.keys(), repeat=k_mer)]
    
    # Generate token to index and index to token dictionaries
    token_to_index = {token: idx for idx, token in enumerate(tokens)}
    index_to_token = {idx: token for idx, token in enumerate(tokens)}
    
    # Write token to index dictionary to file
    filename = f'vocab_{k_mer}mer.txt'
    with open(filename, 'w') as f:
        for token, idx in token_to_index.items():
            f.write(f'{token}\t{idx}\n')
    
    return token_to_index, index_to_token

# Example usage:
if __name__ == "__main__":
    # Generate vocabulary for 1-mers
    token_to_index, index_to_token = generate_vocab(k_mer=2)
    