import numpy as np

data = np.load('bert_vectors.npy')
labels = np.loadtxt('stance_labels.txt', dtype=int)

print(data.shape)
print(labels.shape)
