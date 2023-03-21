# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    #actual true label fraction
    #truelabel_frac = np.sum(labels)/nlabels
    #print(truelabel_frac)

    nlabels = len(labels)

    #separate seqs by label
    seqs_pos = []
    seqs_neg = []

    for x, seq in enumerate(seqs):
        if labels[x]:
            seqs_pos.append(seq)
        else:
            seqs_neg.append(seq)

    #randomly draw a sample consisting of 50% positive and 50% negative examples
    seqs_labels_out = []

    for x in range(nlabels):
        if x%2 == 0:
            seqs_labels_out.append([seqs_neg[np.random.randint(0,len(seqs_neg))],0])
        else:
            seqs_labels_out.append([seqs_pos[np.random.randint(0,len(seqs_pos))],1])

    #shuffle the order of the entries
    np.random.shuffle(seqs_labels_out)

    return ([i[0] for i in seqs_labels_out], [i[1] for i in seqs_labels_out])


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    onehot_dict = {
    "A":[1, 0, 0, 0],
    "T":[0, 1, 0, 0],
    "C":[0, 0, 1, 0],
    "G":[0, 0, 0, 1]}

    return [np.concatenate([onehot_dict[i.upper()] for i in seq]) for seq in seq_arr]