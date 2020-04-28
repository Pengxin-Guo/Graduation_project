import numpy as np
import tensorflow as tf
import image_classification.algorithm.vgg19 as vgg19
import skimage
import skimage.io
import skimage.transform
import image_classification.algorithm.parameter as Para


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def extract_feature_vgg19(img_url):
    with tf.Session() as sess:
        image = tf.placeholder("float", [None, 224, 224, 3])
        sess.run(tf.global_variables_initializer())
        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(image)
        try:
            img = load_image(img_url).reshape((1, 224, 224, 3))
        except ValueError:
            print('There is an error occuring for image: ' + img_url)
        feature = sess.run(vgg.fc7, feed_dict={image: img})
        feature = np.reshape(feature, [1, -1])
        return feature


def np_softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def predict(input, task_id):
    hidden_feature = np.tanh(np.dot(input, Para.input_hidden_weights))
    temp_hidden_feature = np.concatenate([hidden_feature, Para.task_embedding_vectors[task_id].reshape([1, -1])], 1)
    probits_softmax = []
    for j in range(Para.num_class):
        temp = np.concatenate([temp_hidden_feature, Para.class_embedding_vectors[task_id * Para.num_task + j].reshape([1, -1])], 1)
        probit_softmax = np_softmax(np.dot(temp, Para.hidden_output_weight[task_id]))
        probits_softmax.append(probit_softmax)
    probits_softmax = np.squeeze(np.concatenate([probits_softmax], 0))
    diagonal = []
    for j in range(Para.num_class):
        diagonal.append(probits_softmax[j][j])
    class_id = np.argmax(diagonal)
    return class_id, probits_softmax[class_id][class_id]
