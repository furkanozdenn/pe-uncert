import numpy as np
import pandas as pd 
import pdb
import torch

def read_data(path):
    df = pd.read_csv(path)
    return df

def token_to_index_dict(vocab_path):
    token_to_index = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            token, index = line.strip().split('\t')
            token_to_index[token] = int(index)
    return token_to_index

def index_to_token_dict(token_to_index):
    index_to_token = {}
    for token in token_to_index:
        index_to_token[token_to_index[token]] = token
    return index_to_token

def seq_to_inds(seq, token_to_index):
    return np.array([token_to_index[token.upper()] for token in seq]).astype(int)

def df_seq_to_inds(seq_col, token_to_index):
    return seq_col.apply(lambda seq: seq_to_inds(seq, token_to_index))

def inds_to_seq(inds, index_to_token):
    return [index_to_token[ind] for ind in inds]

def df_inds_to_seq(inds_col, index_to_token):
    return inds_col.apply(lambda inds: inds_to_seq(inds, index_to_token))

def dataset_to_data_columns_dict(dataset_name):
    if dataset_name == 'changeseq':
        return {
            'sgrna': 'target',
            'target': 'context sequence flank',
            'activity': 'score'
        }
    elif dataset_name == 'pridict-90k-cleaned':
        return {
            'initial_sequence': 'initial_sequence',
            'mutated_sequence': 'mutated_sequence',
            'total_read_count': 'total_read_count',
            'edited_percentage': 'edited_percentage',
            'unedited_percentage': 'unedited_percentage',
            'indel_percentage': 'indel_percentage',
            'protospacer_location': 'protospacer_location',
            'pbs_location': 'pbs_location',
            'rt_initial_location': 'rt_initial_location',
            'rt_mutated_location': 'rt_mutated_location'
        }
    else:
        raise ValueError(f"dataset_name: {dataset_name} not recognized")
    

def get_binary_location_mask(location_col, max_location):
    """
    the locations format expected is [M,N] (string) where max(N) is max_location
    e.g., '[20,40]' means that the location is from 20 to 40 should be 1, otherwise 0
    """
    binary_masks = []
    for loc in location_col:
        mask = np.zeros(max_location, dtype=int)
        # Remove brackets and split by comma
        start, end = map(int, loc.strip('[]').split(','))
        # Fill 1s from start to end (inclusive)
        mask[start:end+1] = 1
        binary_masks.append(mask)
    return np.array(binary_masks)



'''
Function for one hot encoding n-bp NA sequences
vocab = {A: 0, G: 1, C: 2, T: 3}
each correspond to 4-dim one hot encoding vectors

Args:
seq: str, sequence to one hot encode (should only include chars from vocab) - length n
vocab: dict, vocab mapping from nucleotide to index
Returns:
one_hot_encoded_seq: np.array, shape (n, 4)
'''

def one_hot_encode_seq(seq, vocab = None):
    if vocab is None:
        vocab = {'A': 0, 'G': 1, 'C': 2, 'T': 3, 'N': 4}
    
    one_hot_encoded_seq = np.zeros((len(seq), len(vocab)))
    for i, char in enumerate(seq):
        char = char.upper()
        one_hot_encoded_seq[i, vocab[char]] = 1
    return one_hot_encoded_seq

def decoded_to_seq_(decoded_seq, vocab = None):
    if vocab is None:
        vocab = {0: 'A', 1: 'G', 2: 'C', 3: 'T', 4: 'N'}

    # decoded_seq is np array of (batch_size, 5, 23)
    seqs = []
    for i in range(decoded_seq.shape[0]):
        seq = ''
        for j in range(decoded_seq.shape[2]):
            seq += vocab[np.argmax(decoded_seq[i, :, j])]
        seqs.append(seq)
    
    # return as np array
    return np.array(seqs)

def df_one_hot_encode_seq(seq_col, vocab = None):
    return seq_col.apply(lambda seq: one_hot_encode_seq(seq, vocab))

def decoded_to_seq(decoded_seq, index_to_token):
    """convert models decode output to corresponding sequence

    decode output shape: (n, 5(vocab_size), 23(sgrna_seq_len))
    """
    seq = ''
    for i in range(decoded_seq.shape[0]):
        seq += index_to_token[np.argmax(decoded_seq[i])]
    return seq