import pickle

with open('hierchical_datasets/ImageNet/embeddings_subset.pkl', 'rb') as file:
    data = pickle.load(file)


# print(data['embeddings']['pt_tree1'].shape)

# print(list(data['embeddings'].keys())[:10])
print(data['embeddings'].shape)



import pickle

with open('hierchical_datasets/ImageNet/horopca_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)

# print(data.keys())
# print(data['labels'])
print(data['embeddings'].shape)



import pickle

with open('hierchical_datasets/ImageNet/cosne_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)

# print(data.keys())
# print(data['labels'])

print(data['embeddings'].shape)

print("Finished reading all embedding files.")