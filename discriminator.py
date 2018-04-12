import tensorflow as tf
import os
import numpy as np
from PIL import Image
import sys
from random import shuffle


def open_image(image_path):
    return Image.open(image_path)


def resize(img, width, height):
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img)
    return (img / 255 - 0.5) * 2


use_regularizer = False
batch_size = 32
width = 160
height = 225
num_channels = 3
num_classes = 2
learning_rate = 0.01
num_variation_factor = 1
opt_epsilon = 1.
rmsprop_momentum = 0.9
rmsprop_decay = 0.9
epochs = 10
x_c_idx = 0
y_c_idx = 1
w_idx = 2
h_idx = 3
log_dir = "checkpoint"

inputs_ph = tf.placeholder(tf.float32, [None, height, width, num_channels],
                           "inputs_placeholder")
labels_ph = tf.placeholder(tf.float32, [None, num_classes], "labels_placeholder")
dropout_rate_ph = tf.placeholder(tf.float32, (), "dropout_rate_placeholder")
weight_decay_ph = tf.placeholder(tf.float32, ())
is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

net = tf.layers.conv2d(inputs_ph, 64, 5, 2, padding="VALID", activation=tf.nn.relu,
                       bias_initializer=tf.constant_initializer(0.1))
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu)
net = tf.layers.conv2d(net, 192, 3, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(net, 3, 2)
net = tf.layers.conv2d(net, 2048, [int(net.get_shape()[1]), int(net.get_shape()[2])], padding='VALID',
                       activation=tf.nn.relu)
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, 2048, 1, padding='VALID', activation=tf.nn.relu)
net = tf.layers.dropout(net, dropout_rate_ph, training=is_training_ph)
net = tf.layers.conv2d(net, num_classes, 1)
logits_op = tf.squeeze(net, [1, 2])

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph, logits=logits_op),
                         name="softmax_cross_entropy_with_logits_v2")

if use_regularizer:
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    regularizer = tf.constant(0., tf.float32)
    for weight in weights:
        regularizer += tf.nn.l2_loss(weight)
    regularizer *= weight_decay_ph
    loss_op += regularizer
opt = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=opt_epsilon)
train_op = opt.minimize(loss_op)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_op, 1), tf.argmax(labels_ph, 1)), tf.float32))
pred_idx_op = tf.argmax(logits_op, 1)

softmax_op = tf.nn.softmax(logits_op)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

sess.run(tf.global_variables_initializer())

summary_dir = os.path.join(log_dir, "summary")
train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(summary_dir + '/validation')
saver = tf.train.Saver(tf.global_variables())
import glob

imgs = glob.glob("F:\data\main_image\main_dabainsang\main_image/*")
for i in range(epochs):
    # shuffle(imgs)
    for j in range(0, len(imgs), batch_size):
        files = imgs[j:j + batch_size]
        if len(files) < batch_size:
            break
        y_indices = [1] * (batch_size // 2) + [0] * (batch_size // 2)
        shuffle(y_indices)
        batch_xs = np.array([resize(open_image(file), width, height) for file in files])
        batch_ys = np.array([[1, 0] if k % 2 == 0 else [0, 1] for k, _ in enumerate(files)])

        # im1 = resize(open_image("f:/2.jpg"), width, height)
        # im2 = resize(open_image("f:/1.jpg"), width, height)
        # if i % 2 == 0:
        #     batch_xs = np.array([im1, im2])
        #     batch_ys = np.array([[0, 1], [1, 0]])
        # else:
        #     batch_xs = np.array([im2, im1])
        #     batch_ys = np.array([[1, 0], [0, 1]])
        _, loss, logits, accuracy, pred_idx, softmax = sess.run(
            [train_op, loss_op, logits_op, accuracy_op, pred_idx_op, softmax_op],
            feed_dict={inputs_ph: batch_xs, labels_ph: batch_ys,
                       is_training_ph: True, dropout_rate_ph: 0.})
        print("loss", loss)
        print("accuracy", accuracy)
        print("softmax", softmax)
        print("logits", logits)

        print("pred idx", pred_idx)
