import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes
import numpy.matlib
import re
import os


class MTDataset:
    def __init__(self, data, label, task_interval, num_class, batch_size):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.reshape(label, [1, -1])
        self.task_interval = np.reshape(task_interval, [1, -1])
        self.num_task = task_interval.size - 1
        self.num_class = num_class
        self.batch_size = batch_size
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        for i in range(self.num_task):
            start = self.task_interval[0, i]
            end = self.task_interval[0, i + 1]
            for j in range(self.num_class):
                index_list.append(np.arange(start, end)[np.where(self.label[0, start:end] == j)[0]])
        self.index_list = index_list
        self.counter = np.zeros([1, self.num_task * self.num_class], dtype=np.int32)

    def get_next_batch(self):
        sampled_data = np.zeros([self.batch_size * self.num_class * self.num_task, self.data_dim], dtype=np.float32)
        sampled_label = np.zeros([self.batch_size * self.num_class * self.num_task, self.num_class], dtype=np.int32)
        sampled_task_ind = np.zeros([1, self.batch_size * self.num_class * self.num_task], dtype=np.int32)
        sampled_label_ind = np.zeros([1, self.batch_size * self.num_class * self.num_task], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                sampled_ind = range(cur_ind * self.batch_size, (cur_ind + 1) * self.batch_size)
                sampled_task_ind[0, sampled_ind] = i
                sampled_label_ind[0, sampled_ind] = j
                sampled_label[sampled_ind, j] = 1
                if task_class_index.size < self.batch_size:
                    sampled_data[sampled_ind, :] = self.data[np.concatenate((task_class_index, task_class_index[
                        np.random.randint(0, high=task_class_index.size, size=self.batch_size - task_class_index.size)])), :]
                elif self.counter[0, cur_ind] + self.batch_size < task_class_index.size:
                    sampled_data[sampled_ind, :] = self.data[task_class_index[self.counter[0, cur_ind]:self.counter[0, cur_ind] + self.batch_size], :]
                    self.counter[0, cur_ind] = self.counter[0, cur_ind] + self.batch_size
                else:
                    sampled_data[sampled_ind, :] = self.data[task_class_index[-self.batch_size:], :]
                    self.counter[0, cur_ind] = 0
                    np.random.shuffle(self.index_list[cur_ind])
        return sampled_data, sampled_label, sampled_task_ind, sampled_label_ind


class MTDataset_Split:
    def __init__(self, data, label, task_interval, num_class):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.reshape(label, [1, -1])
        self.task_interval = np.reshape(task_interval, [1, -1])
        self.num_task = task_interval.size - 1
        self.num_class = num_class
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        self.num_class_ins = np.zeros([self.num_task, self.num_class])
        for i in range(self.num_task):
            start = self.task_interval[0, i]
            end = self.task_interval[0, i + 1]
            for j in range(self.num_class):
                index_array = np.where(self.label[0, start:end] == j)[0]
                self.num_class_ins[i, j] = index_array.size
                index_list.append(np.arange(start, end)[index_array])
        self.index_list = index_list

    def split(self, train_size):
        if train_size < 1:
            train_num = np.ceil(self.num_class_ins * train_size).astype(np.int32)
        else:
            train_num = np.ones([self.num_task, self.num_class], dtype=np.int32) * train_size
            train_num = np.maximum(1, np.minimum(train_num, self.num_class_ins - 10))
            train_num = train_num.astype(np.int32)
        traindata = np.zeros([0, self.data_dim], dtype=np.float32)
        testdata = np.zeros([0, self.data_dim], dtype=np.float32)
        trainlabel = np.zeros([1, 0], dtype=np.int32)
        testlabel = np.zeros([1, 0], dtype=np.int32)
        train_task_interval = np.zeros([1, self.num_task + 1], dtype=np.int32)
        test_task_interval = np.zeros([1, self.num_task + 1], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                np.random.shuffle(task_class_index)
                train_index = task_class_index[0:train_num[i, j]]
                test_index = task_class_index[train_num[i, j]:]
                traindata = np.concatenate((traindata, self.data[train_index, :]), axis=0)
                trainlabel = np.concatenate((trainlabel, np.ones([1, train_index.size], dtype=np.int32) * j), axis=1)
                testdata = np.concatenate((testdata, self.data[test_index, :]), axis=0)
                testlabel = np.concatenate((testlabel, np.ones([1, test_index.size], dtype=np.int32) * j), axis=1)
            train_task_interval[0, i + 1] = trainlabel.size
            test_task_interval[0, i + 1] = testlabel.size
        return traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval


def read_data_from_file(filename):
    file = open(filename, 'r')
    contents = file.readlines()
    file.close()
    num_task = int(contents[0])
    num_class = int(contents[1])
    temp_ind = re.split(',', contents[2])
    temp_ind = [int(elem) for elem in temp_ind]
    task_interval = np.reshape(np.array(temp_ind), [1, -1])
    temp_data = []
    for pos in range(3, len(contents) - 1):
        temp_sub_data = re.split(',', contents[pos])
        temp_sub_data = [float(elem) for elem in temp_sub_data]
        temp_data.append(temp_sub_data)
    data = np.array(temp_data)
    temp_label = re.split(',', contents[-1])
    temp_label = [int(elem) for elem in temp_label]
    label = np.reshape(np.array(temp_label), [1, -1])
    return data, label, task_interval, num_task, num_class


def compute_train_loss(i, feature_representation, hidden_output_weight, inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss):
    train_loss += tf.div(tf.losses.softmax_cross_entropy(tf.expand_dims(inputs_data_label[i, :], 0),
                                                         tf.matmul(tf.expand_dims(feature_representation[inputs_task_ind[0, i]][i % (batch_size * inputs_data_label.shape[-1])][:], 0),
                                                                   hidden_output_weight[inputs_task_ind[0, i], :, :])),
                         tf.cast(inputs_num_ins_per_task[0, inputs_task_ind[0, i]], dtype=tf.float32))
    return i + 1, feature_representation, hidden_output_weight, inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss


def gradient_clipping_tf_false_consequence(optimizer, obj, gradient_clipping_threshold):
    gradients, variables = zip(*optimizer.compute_gradients(obj))
    gradients = [None if gradient is None else tf.clip_by_value(gradient, gradient_clipping_threshold,
                                                                tf.negative(gradient_clipping_threshold)) for gradient in gradients]
    train_step = optimizer.apply_gradients(zip(gradients, variables))
    return train_step


def gradient_clipping_tf(optimizer, obj, option, gradient_clipping_threshold):
    train_step = tf.cond(tf.equal(option, 0), lambda: optimizer.minimize(obj),
                         lambda: gradient_clipping_tf_false_consequence(optimizer, obj, gradient_clipping_threshold))
    train_step = tf.group(train_step)
    return train_step


def generate_label_task_ind(label, task_interval, num_class):
    num_task = task_interval.size - 1
    num_ins = label.size
    label_matrix = np.zeros((num_ins, num_class), dtype=np.int32)
    label_matrix[range(num_ins), label] = 1
    task_ind = np.zeros((1, num_ins), dtype=np.int32)
    for i in range(num_task):
        task_ind[0, task_interval[0, i]: task_interval[0, i + 1]] = i
    return label_matrix, task_ind


def compute_errors(hidden_rep, hidden_output_weight, task_ind, label, num_task):
    num_total_ins = hidden_rep.shape[0]
    num_ins = np.zeros([1, num_task])
    errors = np.zeros([1, num_task + 1])
    for i in range(num_total_ins):
        probit = np.matmul(hidden_rep[i, :], hidden_output_weight[task_ind[0, i], :, :])
        num_ins[0, task_ind[0, i]] += 1
        if np.argmax(probit) != label[0, i]:
            errors[0, task_ind[0, i]] += 1
    for i in range(num_task):
        errors[0, i] = errors[0, i] / num_ins[0, i]
    errors[0, num_task] = np.mean(errors[0, 0: num_task])
    return errors


def change_datastruct(hidden_features, num_task):
    return tf.reshape(hidden_features, [num_task, -1, hidden_features.shape[-1]])


def compute_pairwise_dist_tf(data):
    sq_data_norm = tf.reduce_sum(tf.square(data), axis=1)
    sq_data_norm = tf.reshape(sq_data_norm, [-1, 1])
    dist_matrix = sq_data_norm - 2 * tf.matmul(data, data, transpose_b=True) + tf.matrix_transpose(sq_data_norm)
    return dist_matrix


def compute_pairwise_dist_np(data):
    sq_data_norm = np.sum(data ** 2, axis=1)
    sq_data_norm = np.reshape(sq_data_norm, [-1, 1])
    dist_matrix = sq_data_norm - 2 * np.dot(data, data.transpose()) + sq_data_norm.transpose()
    return dist_matrix


def compute_adjacency_matrix(hidden_features, inputs_data_label, num_task):
    new_hidden_features = change_datastruct(hidden_features, num_task)
    new_inputs_data_label = change_datastruct(inputs_data_label, num_task)
    adjacency_matrixs = []
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_tf(new_hidden_features[i])
        sign_matrix = 2 * tf.matmul(new_inputs_data_label[i], tf.matrix_transpose(new_inputs_data_label[i])) - 1
        adjacency_matrix = tf.exp(dist_matrix) * sign_matrix
        adjacency_matrixs.append(adjacency_matrix)
    adjacency_matrixs = tf.stack(adjacency_matrixs)
    return adjacency_matrixs


def activate_function(temp, activate_op):
    if activate_op == 1:
        return tf.tanh(temp)
    elif activate_op == 2:
        return tf.nn.relu(temp)
    elif activate_op == 3:
        return tf.nn.elu(temp)
    else:
        return


def get_normed_distance_tf(data):
    norminator = tf.matmul(data, tf.transpose(data))
    square = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(data), 1)), [norminator.shape[0], 1])
    denorminator = tf.matmul(square, tf.transpose(square))
    return norminator/denorminator


def get_normed_distance_np(data):
    norminator = np.matmul(data, np.transpose(data))
    square = np.reshape(np.sqrt(np.sum(np.square(data), 1)), [norminator.shape[0], 1])
    denorminator = np.matmul(square, np.transpose(square))
    return norminator/denorminator


def GAT(attention_weight, embedding_vectors):
    transformaed_embedding_vectors = tf.matmul(embedding_vectors, attention_weight)
    attention_values = tf.nn.softmax(get_normed_distance_tf(transformaed_embedding_vectors))
    return attention_values


def get_feature_representation(inputs, input_hidden_weights, hidden_features, adjacency_matrix, num_task, num_class, activate_op,
                            first_task_att_w, first_class_att_w, task_attention_weight, class_attention_weight, inputs_data_label):
    hidden_representation = activate_function(
        tf.add(change_datastruct(tf.matmul(inputs, input_hidden_weights), num_task),
               tf.matmul(adjacency_matrix, change_datastruct(hidden_features, num_task))), activate_op)

    new_inputs_data_label = change_datastruct(inputs_data_label, num_task)
    new_adjacency_matrix = []
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_tf(hidden_representation[i])
        sign_matrix = 2 * tf.matmul(new_inputs_data_label[i], tf.matrix_transpose(new_inputs_data_label[i])) - 1
        adjacency_matrix = tf.exp(dist_matrix) * sign_matrix
        new_adjacency_matrix.append(adjacency_matrix)
    new_adjacency_matrix = tf.stack(new_adjacency_matrix)
    new_hidden_representation = activate_function(
        tf.add(change_datastruct(tf.matmul(inputs, input_hidden_weights), num_task),
               tf.matmul(new_adjacency_matrix, hidden_representation)), activate_op)

    task_embedding_vectors = tf.reduce_max(new_hidden_representation, 1)
    task_attention_values = GAT(first_task_att_w, task_embedding_vectors)
    new_task_embedding_vectors = tf.tanh(tf.matmul(task_attention_values, tf.matmul(task_embedding_vectors, first_task_att_w)))
    task_attention_values = GAT(task_attention_weight, new_task_embedding_vectors)
    new_task_embedding_vectors = tf.tanh(tf.matmul(task_attention_values, tf.matmul(new_task_embedding_vectors, task_attention_weight)))

    class_embedding_vectors = []
    for i in range(num_task):
        class_hidden_rep = tf.reshape(new_hidden_representation[i], [num_class, -1, new_hidden_representation.shape[-1]])
        for j in range(num_class):
            class_embedding_vectors.append(tf.reduce_max(class_hidden_rep[j], 0))
    class_embedding_vectors = tf.stack(class_embedding_vectors)
    class_attention_values = GAT(first_class_att_w, class_embedding_vectors)
    new_class_embedding_vectors = tf.tanh(tf.matmul(class_attention_values, tf.matmul(class_embedding_vectors, first_class_att_w)))
    class_attention_values = GAT(class_attention_weight, new_class_embedding_vectors)
    new_class_embedding_vectors = tf.tanh(tf.matmul(class_attention_values, tf.matmul(new_class_embedding_vectors, class_attention_weight)))

    feature_representations = []
    for i in range(num_task):
        feature_representation = []
        feature_representation_1 = tf.concat([
            hidden_features[i * batch_size * num_class: (i + 1) * batch_size * num_class],
            tf.stack([new_task_embedding_vectors[i] for _ in range(num_class * batch_size)])], 1)
        for j in range(num_class):
            feature_representation_2 = tf.concat([
                feature_representation_1[j * batch_size: (j + 1) * batch_size],
                tf.stack([new_class_embedding_vectors[i * num_task + j] for _ in range(batch_size)])], 1)
            feature_representation.append(feature_representation_2)
        feature_representations.append(feature_representation)
    feature_representations = tf.stack(feature_representations)
    return tf.reshape(feature_representations, [num_task, -1, hidden_features.shape[-1] + F_pie_t + F_pie_c])


def np_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_embedding_vec(traindata, input_hidden_weights, first_task_att_w, first_class_att_w, task_attention_weight,
                      class_attention_weight, train_hidden_features, train_label_matrix, train_task_ind, train_num_ins_per_task, num_task,  num_class):
    inputs = [[] for _ in range(num_task)]
    features = [[] for _ in range(num_task)]
    labels = [[] for _ in range(num_task)]
    for i in range(traindata.shape[0]):
        inputs[train_task_ind[0, i]].append(traindata[i])
        features[train_task_ind[0, i]].append(train_hidden_features[i])
        labels[train_task_ind[0, i]].append(train_label_matrix[i])
    task_embedding_vectors = []
    class_embedding_vectors = []
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_np(np.stack(features[i]))
        sign_matrix = 2 * np.matmul(np.stack(labels[i]), np.transpose(np.stack(labels[i]))) - 1
        adjacency_matrix = np.exp(dist_matrix) * sign_matrix
        new_features = np.tanh(np.add(np.matmul(np.stack(inputs[i]), input_hidden_weights),
                                      np.matmul(adjacency_matrix, np.stack(features[i]))))
        new_dist_matrix = -compute_pairwise_dist_np(new_features)
        new_adjacency_matrix = np.exp(new_dist_matrix) * sign_matrix
        new_features = np.tanh(np.add(np.matmul(np.stack(inputs[i]), input_hidden_weights),
                                      np.matmul(new_adjacency_matrix, new_features)))

        task_embedding_vector = np.max(new_features, 0)
        task_embedding_vectors.append(task_embedding_vector)
        inputs_class = [[] for _ in range(num_class)]
        features_class = [[] for _ in range(num_class)]
        labels_class = [[] for _ in range(num_class)]
        for j in range(len(inputs[i])):
            inputs_class[np.int(np.argmax(labels[i][j]))].append(inputs[i][j])
            features_class[np.int(np.argmax(labels[i][j]))].append(features[i][j])
            labels_class[np.int(np.argmax(labels[i][j]))].append(labels[i][j])
        for j in range(num_class):
            dist_matrix = -compute_pairwise_dist_np(np.stack(features_class[j]))
            sign_matrix = 2 * np.matmul(np.stack(labels_class[j]), np.transpose(np.stack(labels_class[j]))) - 1
            adjacency_matrix = np.exp(dist_matrix) * sign_matrix
            new_features = np.tanh(np.add(np.matmul(np.stack(inputs_class[j]), input_hidden_weights),
                                          np.matmul(adjacency_matrix, np.stack(features_class[j]))))
            class_embedding_vector = np.max(new_features, 0)
            class_embedding_vectors.append(class_embedding_vector)
    task_attention_values = np_softmax(get_normed_distance_np(np.stack(task_embedding_vectors)))
    new_task_embedding_vectors = np.tanh(np.matmul(task_attention_values, np.matmul(task_embedding_vectors, first_task_att_w)))
    task_attention_values = np_softmax(get_normed_distance_np(np.stack(new_task_embedding_vectors)))
    new_task_embedding_vectors = np.tanh(np.matmul(task_attention_values, np.matmul(new_task_embedding_vectors, task_attention_weight)))

    class_attention_values = np_softmax(get_normed_distance_np(np.stack(class_embedding_vectors)))
    new_class_embedding_vectors = np.tanh(np.matmul(class_attention_values, np.matmul(class_embedding_vectors, first_class_att_w)))
    class_attention_values = np_softmax(get_normed_distance_np(np.stack(new_class_embedding_vectors)))
    new_class_embedding_vectors = np.tanh(np.matmul(class_attention_values, np.matmul(new_class_embedding_vectors, class_attention_weight)))

    return new_task_embedding_vectors, new_class_embedding_vectors


def get_new_hidden_features(test_hidden_rep, task_embedding_vectors, class_embedding_vectors, hidden_output_weight, test_task_ind, num_task, num_class):
    temp_test_hidden_rep = []
    for i in range(test_hidden_rep.shape[0]):
        temp = np.concatenate([test_hidden_rep[i], task_embedding_vectors[test_task_ind[0, i]]], 0)
        temp_test_hidden_rep.append(temp)
    temp_test_hidden_rep = np.stack(temp_test_hidden_rep)
    test_hidden_rep = []
    for i in range(len(temp_test_hidden_rep)):
        task_id = test_task_ind[0, i]
        probits_softmax = []
        for j in range(num_class):
            temp = np.concatenate([temp_test_hidden_rep[i], class_embedding_vectors[task_id * num_task + j]], 0)
            probit_softmax = np_softmax(np.matmul(temp, hidden_output_weight[task_id]))
            probits_softmax.append(probit_softmax)
        probits_softmax = np.stack(probits_softmax)
        diagonal = []
        for j in range(num_class):
            diagonal.append(probits_softmax[j][j])
        class_id = np.argmax(diagonal)
        test_hidden_rep.append(np.concatenate([temp_test_hidden_rep[i], class_embedding_vectors[task_id * num_class + class_id]], 0))
    test_hidden_rep = np.stack(test_hidden_rep)
    return test_hidden_rep


def DMTL_HGNN(traindata, trainlabel, train_task_interval, dim, num_class, num_task, hidden_dim, batch_size, reg_para,
         max_epoch, testdata, testlabel, test_task_interval, activate_op):
    print('DMTL_HGNN is running...')
    inputs = tf.placeholder(tf.float32, shape=[None, dim])
    inputs_data_label = tf.placeholder(tf.float32, shape=[None, num_class])
    inputs_task_ind = tf.placeholder(tf.int32, shape=[1, None])
    inputs_num_ins_per_task = tf.placeholder(tf.int32, shape=[1, None])
    input_hidden_weights = tf.Variable(tf.truncated_normal([dim, hidden_dim], dtype=tf.float32, stddev=1e-1))
    hidden_features = activate_function(tf.matmul(inputs, input_hidden_weights), activate_op)
    adjacency_matrix = compute_adjacency_matrix(hidden_features, inputs_data_label, num_task)

    first_task_att_w = tf.Variable(tf.truncated_normal(
        [hidden_dim, GAT_hidden_dim], dtype=tf.float32, stddev=1e-1))
    first_class_att_w = tf.Variable(tf.truncated_normal(
        [hidden_dim, GAT_hidden_dim], dtype=tf.float32, stddev=1e-1))
    task_attention_weight = tf.Variable(tf.truncated_normal(
        [GAT_hidden_dim, F_pie_t], dtype=tf.float32, stddev=1e-1))
    class_attention_weight = tf.Variable(tf.truncated_normal(
        [GAT_hidden_dim, F_pie_c], dtype=tf.float32, stddev=1e-1))

    feature_representation = get_feature_representation(inputs, input_hidden_weights, hidden_features, adjacency_matrix,
                                               num_task, num_class, activate_op, first_task_att_w, first_class_att_w, task_attention_weight, class_attention_weight, inputs_data_label)

    hidden_output_weight = tf.Variable(tf.truncated_normal(
        [num_task, hidden_dim + F_pie_t + F_pie_c, num_class], dtype=tf.float32, stddev=1e-1))

    train_loss = tf.Variable(0.0, dtype=tf.float32)
    _, _, _, _, _, _, train_loss = tf.while_loop(
        cond=lambda i, j1, j2, j3, j4, j5, j6: tf.less(i, tf.shape(inputs_task_ind)[1]), body=compute_train_loss,
        loop_vars=(tf.constant(0, dtype=tf.int32), feature_representation, hidden_output_weight,
                   inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss))

    obj = train_loss + reg_para * (tf.square(tf.norm(input_hidden_weights))+tf.square(tf.norm(hidden_output_weight)))

    learning_rate = tf.placeholder(tf.float32)
    gradient_clipping_threshold = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradient_clipping_option = tf.placeholder(tf.int32)
    train_step = gradient_clipping_tf(optimizer, obj, gradient_clipping_option, gradient_clipping_threshold)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        max_iter_epoch = numpy.ceil(traindata.shape[0] / (batch_size * num_task * num_class)).astype(
            np.int32)
        Iterator = MTDataset(traindata, trainlabel, train_task_interval, num_class, batch_size)
        sess.run(init_op)

        train_label_matrix, train_task_ind = generate_label_task_ind(trainlabel, train_task_interval, num_class)
        for iter in range(max_iter_epoch * max_epoch):
            sampled_data, sampled_label, sampled_task_ind, _ = Iterator.get_next_batch()
            num_iter = iter // max_iter_epoch
            train_step.run(feed_dict={d1: d2 for d1, d2 in
                                      zip([learning_rate, gradient_clipping_option, gradient_clipping_threshold, inputs,
                                           inputs_data_label, inputs_task_ind, inputs_num_ins_per_task],
                                          [0.02 / (1 + num_iter), 0, -5., sampled_data, sampled_label, sampled_task_ind,
                                           np.ones([1, num_task]) * (batch_size * num_class)])})
            if iter % max_iter_epoch == 0 and num_iter % 5 == 0:
                train_hidden_features = hidden_features.eval(feed_dict={inputs: traindata, inputs_task_ind: train_task_ind})
                task_embedding_vectors, class_embedding_vectors = get_embedding_vec(traindata, input_hidden_weights.eval(), first_task_att_w.eval(), first_class_att_w.eval(), task_attention_weight.eval(), class_attention_weight.eval(),
                                    train_hidden_features, train_label_matrix, train_task_ind, np.reshape(
                                   train_task_interval[0, 1:] - train_task_interval[0, 0:num_task], [1, -1]), num_task, num_class)
                _, test_task_ind = generate_label_task_ind(testlabel, test_task_interval, num_class)
                test_hidden_rep = hidden_features.eval(feed_dict={inputs: testdata, inputs_task_ind: test_task_ind})
                new_test_hidden_rep = get_new_hidden_features(test_hidden_rep, task_embedding_vectors, class_embedding_vectors, hidden_output_weight.eval(), test_task_ind, num_task, num_class)
                test_errors = compute_errors(new_test_hidden_rep, hidden_output_weight.eval(), test_task_ind, testlabel,
                                                 num_task)
                print('epoch = %g, test_errors = %s' % (num_iter, test_errors))
    return test_errors


def main_process(filename, train_size, hidden_dim, batch_size, reg_para, max_epoch, use_gpu, gpu_id='0', activate_op=1):
    if use_gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    data, label, task_interval, num_task, num_class = read_data_from_file(filename)
    data_split = MTDataset_Split(data, label, task_interval, num_class)
    dim = data.shape[1]
    traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval = data_split.split(train_size)
    error = DMTL_HGNN(traindata, trainlabel, train_task_interval, dim, num_class, num_task, hidden_dim,
                 batch_size, reg_para, max_epoch, testdata, testlabel, test_task_interval, activate_op)
    return error


datafile = './data/office_caltech_10_fc7.txt'
max_epoch = 200
use_gpu = 1
gpu_id = '2'
hidden_dim = 600
batch_size = 32
reg_para = 0.2
train_size = 0.7
activate_op = 1
GAT_hidden_dim = 16
F_pie_t = 8
F_pie_c = 8

mean_errors = main_process(datafile, train_size, hidden_dim, batch_size, reg_para, max_epoch, use_gpu, gpu_id,
                           activate_op)

print('final test_errors = ', mean_errors)
