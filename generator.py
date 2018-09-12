# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot


class Generator:
    def __init__(self, inputs, is_training, input_height, input_width, real_height, real_width, num_classes=10,
                 mode=1):
        self.bg_transform = None
        self.title_transform = None
        self.credit_transform = None
        with tf.variable_scope("generator"):
            print(inputs)
            if mode == 1:
                bg_transform, title_transform, credit_transform = self.transform_mixed(inputs, real_height, real_width)
            elif mode == 2:
                bg_transform, title_transform, credit_transform = self.transform_isotropic(inputs, real_height,
                                                                                           real_width)
            else:
                bg, title, credit = tf.split(inputs, [3, 4, 4], 3)

                tf.summary.image("bg", bg, max_outputs=10)
                tf.summary.image("title", title, max_outputs=10)
                tf.summary.image("credit", credit, max_outputs=10)
                bg_transform = self.transform(bg, real_height, real_width)
                title_transform = self.transform(title, real_height, real_width)
                credit_transform = self.transform(credit, real_height, real_width)
            self.bg_transform = bg_transform
            self.title_transform = title_transform
            self.credit_transform = credit_transform
            tf.summary.image("bg_transform", bg_transform, max_outputs=10)
            tf.summary.image("title_transform", title_transform, max_outputs=10)
            tf.summary.image("credit_transform", credit_transform, max_outputs=10)

            title_rgb, title_a = tf.split(title_transform, [3, 1], 3)
            title_rgb = title_rgb * title_a
            title_a_reverse = 1 - title_a

            credit_rgb, credit_a = tf.split(credit_transform, [3, 1], 3)
            credit_rgb = credit_rgb * credit_a
            credit_a_reverse = 1 - credit_a

            bg_transform = bg_transform * title_a_reverse
            bg_transform = bg_transform + title_rgb

            bg_transform = bg_transform * credit_a_reverse
            bg_transform = bg_transform + credit_rgb

            self.fake_image = bg_transform

    def transform(self, inputs, out_height, out_width):
        net = tf.layers.conv2d(inputs, 64, 7, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.reduce_mean(net, [1, 2], name='global_pool')
        net = tf.layers.dense(net, 20, activation=tf.nn.tanh)
        net = tf.layers.dropout(net, 0.8)
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        net = tf.layers.dense(net, 6, activation=tf.nn.tanh, bias_initializer=tf.initializers.constant(initial))

        return transformer(inputs, net, (out_height, out_width))

    def transform_mixed(self, inputs, out_height, out_width):
        net = tf.layers.conv2d(inputs, 64, 7, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.reduce_mean(net, [1, 2], name='global_pool')
        net = tf.layers.dense(net, 36, activation=tf.nn.tanh)
        net = tf.layers.dropout(net, 0.8)
        initial = np.array([[1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        net = tf.layers.dense(net, 18, activation=tf.nn.tanh, bias_initializer=tf.initializers.constant(initial))

        bg, title, credit = tf.split(inputs, [3, 4, 4], 3)
        tf.summary.image("bg", bg, max_outputs=10)
        tf.summary.image("title", title, max_outputs=10)
        tf.summary.image("credit", credit, max_outputs=10)
        bg_p, title_p, credit_p = tf.split(net, 3, 1)
        bg_trans = transformer(bg, bg_p, (out_height, out_width))
        title_trans = transformer(title, title_p, (out_height, out_width))
        credit_trans = transformer(credit, credit_p, (out_height, out_width))
        return bg_trans, title_trans, credit_trans

    def transform_isotropic(self, inputs, out_height, out_width):
        net = tf.layers.conv2d(inputs, 64, 7, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net, 3, 2)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 256, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.layers.conv2d(net, 128, 3, activation=tf.nn.relu)
        net = tf.reduce_mean(net, [1, 2], name='global_pool')
        net = tf.layers.dense(net, 36, activation=tf.nn.tanh)
        net = tf.layers.dropout(net, 0.8)
        initial = np.array([[1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        net = tf.layers.dense(net, 18, activation=tf.nn.tanh, bias_initializer=tf.initializers.constant(initial))

        bg, title, credit = tf.split(inputs, [3, 4, 4], 3)
        tf.summary.image("bg", bg, max_outputs=10)
        tf.summary.image("title", title, max_outputs=10)
        tf.summary.image("credit", credit, max_outputs=10)
        bg_p, title_p, credit_p = tf.split(net, 3, 1)
        bg_p[1] = 0
        bg_p[3] = 0
        bg_p[4] = bg_p[0]
        title_p[1] = 0
        title_p[3] = 0
        title_p[4] = title_p[0]
        credit_p[1] = 0
        credit_p[3] = 0
        credit_p[4] = credit_p[0]
        bg_trans = transformer(bg, bg_p, (out_height, out_width))
        title_trans = transformer(title, title_p, (out_height, out_width))
        credit_trans = transformer(credit, credit_p, (out_height, out_width))
        return bg_trans, title_trans, credit_trans

    def generate(self):
        return self.fake_image
        # return tf.reshape(self.fake_image,
        #                   [1, int(self.fake_image.get_shape()[0]), int(self.fake_image.get_shape()[1]), 3])
