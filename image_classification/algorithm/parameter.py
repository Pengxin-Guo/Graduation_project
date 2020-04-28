import numpy as np

num_task = 4
num_class = 10


input_hidden_weights = np.load('./parameter/input_hidden_weights.npy')
hidden_output_weight = np.load('./parameter/hidden_output_weight.npy')
task_embedding_vectors = np.load('./parameter/task_embedding_vectors.npy')
class_embedding_vectors = np.load('./parameter/class_embedding_vectors.npy')
