import numpy as np
import Graduation_project.settings as settings

num_task = 4
num_class = 10

path = settings.PARA_PATH
input_hidden_weights = np.load(path + 'input_hidden_weights.npy')
hidden_output_weight = np.load(path + 'hidden_output_weight.npy')
task_embedding_vectors = np.load(path + 'task_embedding_vectors.npy')
class_embedding_vectors = np.load(path + 'class_embedding_vectors.npy')
