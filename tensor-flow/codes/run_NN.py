import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings_list_np = np.loadtxt('embeddings.txt')
mappings = np.load('mappings.npy') 
n_neighbors = 5
nnbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(embeddings_list_np)
distances, neighbors_indices = nnbrs.kneighbors(embeddings_list_np)
print(neighbors_indices)