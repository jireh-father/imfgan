import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys


def open_image(image_path):
    return Image.open(image_path)


def resize(img, size):
    img = img.resize((size, size), Image.ANTIALIAS)
    return np.array(img)


use_regularizer = False
batch_size = 1
input_size = 160
padding_input_size = 240
num_channel = 4
num_input_images = 3
learning_rate = 0.01
num_variation_factor = 1
opt_epsilon = 1.
rmsprop_momentum = 0.9
rmsprop_decay = 0.9
steps = 100
x_c_idx = 0
y_c_idx = 1
w_idx = 2
h_idx = 3
log_dir = "checkpoint"

inputs_ph = tf.placeholder(tf.float32, [batch_size, input_size, input_size, num_channel * num_input_images],
                           "inputs_placeholder")
labels_ph = tf.placeholder(tf.float32, [batch_size, num_variation_factor * num_input_images], "labels_placeholder")
dropout_rate_ph = tf.placeholder(tf.float32, (), "dropout_rate_placeholder")
weight_decay_ph = tf.placeholder(tf.float32, ())
is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

net = tf.layers.conv2d(inputs_ph, 64, 5, 2, padding="VALID", activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 2048, [int(net.get_shape()[1]), int(net.get_shape()[2])], padding='VALID',
                       activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, 2048, 1, padding='VALID', activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, num_variation_factor * num_input_images, 1,
                       bias_initializer=tf.zeros_initializer())
logits_op = tf.squeeze(net, [1, 2])
sigmoid_op = tf.nn.sigmoid(logits_op)
input_images = tf.split(tf.reshape(inputs_ph, [input_size, input_size, num_channel * num_input_images]),
                        num_input_images, axis=2)

merged_image = tf.constant(0., tf.float32, shape=[input_size, input_size, num_channel])
for i in range(len(input_images)):
    width_idx = w_idx * (i * num_variation_factor)
    width = (sigmoid_op[width_idx] + 0.5) * padding_input_size
    height_idx = h_idx * (i * num_variation_factor)
    height = (sigmoid_op[i] + 0.5) * padding_input_size

    x_coord_idx = x_c_idx * (i * num_variation_factor)
    x_coord = sigmoid_op[x_coord_idx] * padding_input_size
    y_coord_idx = y_c_idx * (i * num_variation_factor)
    y_coord = +sigmoid_op[y_coord_idx] * padding_input_size

    tf.image.resize_images(input_images, [])

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_ph, logits=logits_op),
                         name="sigmoid_cross_entropy")

if use_regularizer:
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    regularizer = tf.constant(0., tf.float32)
    for weight in weights:
        regularizer += tf.nn.l2_loss(weight)
    regularizer *= weight_decay_ph
    loss_op += regularizer
opt = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=opt_epsilon)
train_op = opt.minimize(loss_op)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

sess.run(tf.global_variables_initializer())

summary_dir = os.path.join(log_dir, "summary")
train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(summary_dir + '/validation')
saver = tf.train.Saver(tf.global_variables())

im1 = resize(open_image("f:/2.jpg"), input_size)
im2 = resize(open_image("f:/1.jpg"), input_size)
im3 = np.concatenate((im1, im2), axis=2)

batch_xs = np.array([im3])
batch_ys = np.array([[0.5, 0.3, 0.2, 0.9, 0.1, 0.4, 0.12, 0.47]])

for i in range(steps):
    _, loss, logits, sigmoid = sess.run([train_op, loss_op, logits_op, sigmoid_op],
                                        feed_dict={inputs_ph: batch_xs, labels_ph: batch_ys,
                                                   is_training_ph: True, dropout_rate_ph: 0.})
    print(loss)
    print(logits)
    print(sigmoid)
