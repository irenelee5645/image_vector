import numpy as np

import torch
import pickle as pkl

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='utf-8')
    return dict

# def reduce_to_k_dim(M, k=2):
#     """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
#         to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
#             - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
#         Params:
#             M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurence matrix of word counts
#             k (int): embedding size of each word after dimension reduction
#         Return:
#             M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
#                     In terms of the SVD from math class, this actually returns U * S
#     """    
#     n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
#     M_reduced = None
    
#     svd = TruncatedSVD(n_components=k, n_iter=n_iters)
#     #Must call fit for the appropriate matrix before calling transform
#     svd.fit(M)
#     M_reduced = svd.transform(M)


#     print("Done.")
#     return M_reduced