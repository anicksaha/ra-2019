import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings_list_np = np.loadtxt('embeddings.txt')
mappings = np.load('mappings.npy') 
n_neighbors = 5
nnbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(embeddings_list_np)
distances, neighbors_indices = nnbrs.kneighbors(embeddings_list_np)
print(neighbors_indices)

# scores = 1 - (distances / np.max(distances))
# #print(scores)
# #Since the first neighbor is itself, the first column will correspond to input image
# n_rows = 1
# normalized_reduced_examples_visualized_indices = np.random.randint(low=0, high=20, size=(n_rows))
# embedding = np.empty((n_rows * n_neighbors, 2))
# embedding_filenames = [None] * (n_rows * n_neighbors)

# for row in range(n_rows):
#     example_index = normalized_reduced_examples_visualized_indices[row]
#     example_neighbors = neighbors_indices[example_index]
#     for column in range(n_neighbors):
#         i = row * n_neighbors + column
#         embedding[i, 0] = column
#         embedding[i, 1] = row
#         embedding_filenames[i] = imagenames_list[example_neighbors[column]]

# print(embedding_filenames)
# #check if weights are proper
# #randomized index of image use map
# #gray scale
#open face, VGG face net50