# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot


class Generator:
    def __init__(self, inputs, is_training, input_height, input_width, real_height, real_width, num_classes=10):
        def kernel_initializer():
            return tf.truncated_normal_initializer(0.0, 0.005)

        def bias_initializer():
            return tf.constant_initializer(0.1)

        def kernel_regularizer():
            return tf.contrib.layers.l2_regularizer(0.00004)

        with tf.variable_scope("generator"):
            print(inputs)
            bg, title, credit = tf.split(inputs, [3, 4, 4], 3)
            bg_transform = self.transform(bg, real_height, real_width)
            title_transform = self.transform(title, real_height, real_width)
            credit_transform = self.transform(credit, real_height, real_width)

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

            # # net = tf.layers.conv2d(inputs, 64, [11, 11], 4, padding='VALID', name='conv1', activation=tf.nn.relu,
            # #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            # net = tf.layers.conv2d(inputs, 64, [5, 5], 2, padding='SAME', name='conv1', activation=tf.nn.relu,
            #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            # # net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool1')
            #
            # net = tf.layers.conv2d(net, 192, [3, 3], name='conv2', padding='SAME', activation=tf.nn.relu,
            #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            # # net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool2')
            #
            # net = tf.layers.conv2d(net, 384, [3, 3], name='conv3', padding='SAME', activation=tf.nn.relu,
            #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            # net = tf.layers.conv2d(net, 384, [3, 3], name='conv4', padding='SAME', activation=tf.nn.relu,
            #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            #
            # net = tf.layers.conv2d(net, 256, [3, 3], name='conv5', padding='SAME', activation=tf.nn.relu,
            #                        bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            #
            # logits = tf.layers.conv2d(net, 2, [3, 3], name='conv6', padding='SAME', activation=tf.nn.sigmoid,
            #                           bias_initializer=bias_initializer(), kernel_regularizer=kernel_regularizer())
            # # print(net)
            # # sys.exit()
            # # net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool5')
            # # net = tf.layers.conv2d(net, 4096, [int(net.get_shape()[1]), int(net.get_shape()[2])], padding='VALID',
            # #                        name='fc6', activation=tf.nn.relu,
            # #                        kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer(),
            # #                        kernel_regularizer=kernel_regularizer())
            # # net = tf.layers.dropout(net, 0.5, training=is_training, name='dropout6')
            # # net = tf.layers.conv2d(net, 4096, [1, 1], name='fc7', activation=tf.nn.relu, padding='SAME',
            # #                        kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer(),
            # #                        kernel_regularizer=kernel_regularizer())
            # # net = tf.layers.dropout(net, 0.5, training=is_training, name='dropout7')
            # #
            # # net = tf.layers.conv2d(net, num_classes, [1, 1], kernel_initializer=kernel_initializer(),
            # #                        bias_initializer=tf.zeros_initializer(), name='fc8', padding='SAME',
            # #                        kernel_regularizer=kernel_regularizer())
            # # logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            # self._build_fake_image(inputs, tf.nn.softmax(logits), input_height, input_width, real_height, real_width)

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

        # W_fc_loc1 = weight_variable([1600, 20])
        # b_fc_loc1 = bias_variable([20])
        #
        # W_fc_loc2 = weight_variable([20, 6])
        # # Use identity transformation as starting point
        # initial = np.array([[1., 0, 0], [0, 1., 0]])
        # initial = initial.astype('float32')
        # initial = initial.flatten()
        # b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
        # tf.layers.dense(bias_initializer=tf.initializers.constant(initial))
        #
        # # %% Define the two layer localisation network
        # h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
        # # %% We can add dropout for regularizing and to reduce overfitting like so:
        # keep_prob = tf.placeholder(tf.float32)
        # h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
        # # %% Second layer
        # h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
        #
        # # %% We'll create a spatial transformer module to identify discriminative
        # # %% patches
        return transformer(inputs, net, (out_height, out_width))

    def _build_fake_image(self, inputs, logits, input_height, input_width, output_height, output_width):
        bg, title, credit = tf.split(inputs, [3, 4, 4], 3)
        title_logits, credit_logits = tf.split(logits, 2, 3)
        print(title_logits)
        title_start = tf.cast(tf.argmax(
            tf.reshape(title_logits, [int(title_logits.get_shape()[1]) * int(title_logits.get_shape()[2])])), tf.int32)
        title_start_y = tf.cast(title_start / int(title_logits.get_shape()[2]), tf.int32)
        title_start_x = tf.cast(title_start - title_start_y * int(title_logits.get_shape()[2]), tf.int32)
        print(title_start_y)
        print(title_start_x)
        sys.exit()

        bg_x = tf.cast(logits[0][0] * (input_width - output_width), tf.int32)
        bg_y = tf.cast(logits[0][1] * (input_height - output_height), tf.int32)
        bg = tf.image.crop_to_bounding_box(bg, bg_y, bg_x, output_height, output_width)

        title_w = tf.cast(logits[0][4] * (output_width - 1) + 1, tf.int32)
        title_h = tf.cast(logits[0][5] * (output_height - 1) + 1, tf.int32)
        title_x = tf.cast(logits[0][2] * output_width, tf.int32)
        title_y = tf.cast(logits[0][3] * output_height, tf.int32)
        title = tf.image.resize_images(title, (title_h, title_w))
        title_crop_w = title_w - tf.nn.relu(title_x + title_w - output_width)
        title_crop_h = title_h - tf.nn.relu(title_y + title_h - output_height)
        title = tf.image.crop_to_bounding_box(title, 0, 0, output_height, output_width)
        title, _ = tf.split(title, [3, 1], 3)
        logits = tf.pad(logits, tf.constant([[0, 0], [44995, 44995]]), "REFLECT")
        print(logits)
        self.fake_image = (bg + title)  # * logits[0][0]
        print("fake", self.fake_image)
        return
        # 1. 배경
        # 2. 텍스트
        # (1) 배경 리사이징
        logits = tf.image.resize_images(logits, (output_height, output_width))
        print(logits)
        sys.exit()
        idx = tf.where(logits > 0.5)
        _, idx_h, idx_w, idx_c = tf.split(idx, 4, 1)
        y_max = tf.maximum(idx_h)
        y_min = tf.minimum(idx_h)
        x_max = tf.maximum(idx_w)
        x_min = tf.minimum(idx_w)

        print(idx)
        sys.exit()
        bg, title, credit = tf.split(inputs, [3, 4, 4], 3)
        # bg, title, credit [batch, h, w, c]
        # logits [batch, 10]
        bg_x = tf.cast(logits[0][0] * (input_width - output_width), tf.int32)
        bg_y = tf.cast(logits[0][1] * (input_height - output_height), tf.int32)
        print(bg, bg_y, bg_x, output_height, output_width)
        bg = tf.image.crop_to_bounding_box(bg, bg_y, bg_x, tf.convert_to_tensor(output_height, tf.int32),
                                           tf.convert_to_tensor(output_width, tf.int32))

        # title resize
        title_w = tf.cast(logits[0][4] * (output_width - 1) + 1, tf.int32)
        title_h = tf.cast(logits[0][5] * (output_height - 1) + 1, tf.int32)
        title_x = tf.cast(logits[0][2] * output_width, tf.int32)
        title_y = tf.cast(logits[0][3] * output_height, tf.int32)
        title = tf.image.resize_images(title, (title_h, title_w))
        title_crop_w = title_w - tf.nn.relu(title_x + title_w - output_width)
        title_crop_h = title_h - tf.nn.relu(title_y + title_h - output_height)
        title = tf.image.crop_to_bounding_box(title, 0, 0, title_crop_h, title_crop_w)

        # 3. (2) 텍스트 rgb 분리
        # 4. (2) 텍스트 알파값 분리
        title_rgb, title_a = tf.split(title, [3, 1], 3)

        # (4) 알파값 노말라이즈
        title_a = (title_a - tf.reduce_min(title_a)) / tf.reduce_max(title_a) - tf.reduce_min(title_a)

        # (3) 텍스트 rgb * (4) 알파값
        title_rgb = title_rgb * title_a

        # (3) 텍스트 rgb padding with 0 constant
        print(title_rgb)
        print(title_y, output_height, title_y, title_crop_h)
        # title_rgb_paddings = tf.Variable(
        #     [[0, 0], [title_y, output_height - (title_y + title_crop_h)], [title_x, output_width - (
        #       title_x + title_crop_w)], [0, 0]], trainable=False)
        title_rgb_paddings = tf.constant([[0, 0], [10, 10], [10, 10], [0, 0]])
        title_rgb = tf.pad(title_rgb, title_rgb_paddings, constant_values=0.)
        # 5. 알파값 리버스 = 1 - (4) 알파값
        title_a_reverse = 1 - title_a
        # (5) 알파값 (1) 배경 사이즈로 padding with 1 constant
        title_a_reverse = tf.pad(title_a_reverse, title_rgb_paddings, constant_values=1.)
        # (1) 배경 *= (5) 알파값 리버스
        bg = bg * title_a_reverse
        # (1) 배경 += (3) 텍스트 rgb
        bg = bg + title_rgb
        self.fake_image = bg
        print(bg)
        return

        # output_r = tf.Variable(tf.zeros([1, output_height, output_width, 3], tf.float32), trainable=False)
        #
        # inputs, _, _ = tf.split(inputs, [3, 4, 4], 3)
        # logits = tf.gather(logits[0], [0])
        # inputs = inputs * logits[0]
        #
        # inputs = tf.image.crop_to_bounding_box(inputs, 0, 0, output_height + 10, output_width + 10)
        # inputs = tf.image.resize_images(inputs, (output_height, output_width))
        #
        # indices = tf.where(tf.equal(inputs, 1.0))
        # indices = tf.cast(indices, dtype=tf.int32)
        # print(output_r)
        # print(indices)
        # # self.fake_image = inputs + output_r[0][indices[0][0]][0][0]
        # # a = tf.concat(axis=0, values=[a[:i], [updated_value], a[i + 1:]])
        # output_r = tf.scatter_mul(output_r, indices, tf.gather(inputs, indices))
        # self.fake_image = output_r * inputs
        # return
        # min_color_val = -1.
        # output_r = tf.Variable(tf.zeros([output_width * output_height], tf.float32), trainable=False)
        # output_g = tf.Variable(tf.zeros([output_width * output_height], tf.float32), trainable=False)
        # output_b = tf.Variable(tf.zeros([output_width * output_height], tf.float32), trainable=False)
        #
        # split0, split1, split2 = tf.split(inputs, [3, 4, 4], 3)
        # # print(split0, split1, split2)
        #
        # split0 = tf.squeeze(split0)
        # split1 = tf.squeeze(split1)
        # split2 = tf.squeeze(split2)
        #
        # # handle bg manifold
        # bg_manifold = tf.gather(logits[0], [0, 1])
        # bg_x = tf.cast(bg_manifold[0] * (input_width - output_width), tf.int32)
        # bg_y = tf.cast(bg_manifold[1] * (input_height - output_height), tf.int32)
        # bg_img = tf.image.crop_to_bounding_box(split0, bg_y, bg_x, output_height, output_width)
        #
        # # handle title manifold
        # credit_manifold = tf.gather(logits[0], [2, 3, 4, 5])
        # title_w = tf.cast(credit_manifold[2] * (output_width - 1) + 1, tf.int32)
        # title_h = tf.cast(credit_manifold[3] * (output_height - 1) + 1, tf.int32)
        # title_x = tf.cast(credit_manifold[0] * output_width, tf.int32)
        # title_y = tf.cast(credit_manifold[1] * output_height, tf.int32)
        # split1 = tf.image.resize_images(split1, (title_h, title_w))
        # title_crop_w = title_w - tf.nn.relu(title_x + title_w - output_width)
        # title_crop_h = title_h - tf.nn.relu(title_y + title_h - output_height)
        # title_img = tf.image.crop_to_bounding_box(split1, 0, 0, title_crop_h, title_crop_w)
        #
        # # handle credit manifold
        # credit_manifold = tf.gather(logits[0], [6, 7, 8, 9])
        # credit_w = tf.cast(credit_manifold[2] * (output_width - 1) + 1, tf.int32)
        # credit_h = tf.cast(credit_manifold[3] * (output_height - 1) + 1, tf.int32)
        # credit_x = tf.cast(credit_manifold[0] * output_width, tf.int32)
        # credit_y = tf.cast(credit_manifold[1] * output_height, tf.int32)
        # split2 = tf.image.resize_images(split2, (credit_h, credit_w))
        # credit_crop_w = credit_w - tf.nn.relu(credit_x + credit_w - output_width)
        # credit_crop_h = credit_h - tf.nn.relu(credit_y + credit_h - output_height)
        # credit_img = tf.image.crop_to_bounding_box(split2, 0, 0, credit_crop_h, credit_crop_w)
        #
        # # split bg by channel
        # bg_r, bg_g, bg_b = tf.split(bg_img, 3, 2)
        # bg_r = tf.reshape(bg_r, [-1])
        # bg_g = tf.reshape(bg_g, [-1])
        # bg_b = tf.reshape(bg_b, [-1])
        #
        # output_r = output_r.assign_add(bg_r)
        # output_g = output_g.assign_add(bg_g)
        # output_b = output_b.assign_add(bg_b)
        #
        # # lay title over bg!
        # title_r, title_g, title_b, title_alpha = tf.split(title_img, 4, 2)
        #
        # title_alpha = (title_alpha - tf.reduce_min(title_alpha)) / tf.reduce_max(title_alpha) - tf.reduce_min(
        #     title_alpha)
        #
        # title_r = tf.reshape(title_r * title_alpha, [-1])
        # title_g = tf.reshape(title_g * title_alpha, [-1])
        # title_b = tf.reshape(title_b * title_alpha, [-1])
        # title_a = tf.reshape(1 - title_alpha, [-1])
        #
        # title_alpha_1d = tf.reshape(title_alpha, [-1])
        # title_tmp_indices = tf.where(tf.not_equal(title_alpha_1d, min_color_val))
        # title_tmp_indices = tf.cast(title_tmp_indices, dtype=tf.int32)
        #
        # title_r_update = tf.reshape(tf.gather(title_r, title_tmp_indices), [-1])
        # title_g_update = tf.reshape(tf.gather(title_g, title_tmp_indices), [-1])
        # title_b_update = tf.reshape(tf.gather(title_b, title_tmp_indices), [-1])
        # title_a_update = tf.reshape(tf.gather(title_a, title_tmp_indices), [-1])
        #
        # title_color_indices = tf.where(tf.not_equal(title_alpha, min_color_val))
        # title_color_indices = tf.cast(title_color_indices, dtype=tf.int32)
        # title_color_indices = title_color_indices + tf.Variable([title_y, title_x, 0], dtype=tf.int32, trainable=False)
        # title_color_indices1, title_color_indices2, title_color_indices3 = tf.split(title_color_indices, 3, axis=1)
        # output_indices = title_color_indices1 * output_width + title_color_indices2
        #
        # output_indices = tf.reshape(output_indices, [-1])
        #
        # output_r = tf.scatter_mul(output_r, output_indices, title_a_update)
        # output_g = tf.scatter_mul(output_g, output_indices, title_a_update)
        # output_b = tf.scatter_mul(output_b, output_indices, title_a_update)
        #
        # output_r = tf.scatter_add(output_r, output_indices, title_r_update)
        # output_g = tf.scatter_add(output_g, output_indices, title_g_update)
        # output_b = tf.scatter_add(output_b, output_indices, title_b_update)
        #
        # # lay credit over bg!
        # credit_r, credit_g, credit_b, credit_alpha = tf.split(credit_img, 4, 2)
        #
        # credit_alpha = (credit_alpha - tf.reduce_min(credit_alpha)) / tf.reduce_max(credit_alpha) - tf.reduce_min(
        #     credit_alpha)
        #
        # credit_r = tf.reshape(credit_r * credit_alpha, [-1])
        # credit_g = tf.reshape(credit_g * credit_alpha, [-1])
        # credit_b = tf.reshape(credit_b * credit_alpha, [-1])
        # credit_a = tf.reshape(1 - credit_alpha, [-1])
        #
        # credit_alpha_1d = tf.reshape(credit_alpha, [-1])
        # credit_tmp_indices = tf.where(tf.not_equal(credit_alpha_1d, min_color_val))
        # credit_tmp_indices = tf.cast(credit_tmp_indices, dtype=tf.int32)
        #
        # credit_r_update = tf.reshape(tf.gather(credit_r, credit_tmp_indices), [-1])
        # credit_g_update = tf.reshape(tf.gather(credit_g, credit_tmp_indices), [-1])
        # credit_b_update = tf.reshape(tf.gather(credit_b, credit_tmp_indices), [-1])
        # credit_a_update = tf.reshape(tf.gather(credit_a, credit_tmp_indices), [-1])
        #
        # credit_color_indices = tf.where(tf.not_equal(credit_alpha, min_color_val))
        # credit_color_indices = tf.cast(credit_color_indices, dtype=tf.int32)
        # credit_color_indices = credit_color_indices + tf.Variable([credit_y, credit_x, 0], dtype=tf.int32,
        #                                                           trainable=False)
        # credit_color_indices1, credit_color_indices2, credit_color_indices3 = tf.split(credit_color_indices, 3, axis=1)
        # output_indices = credit_color_indices1 * output_width + credit_color_indices2
        #
        # output_indices = tf.reshape(output_indices, [-1])
        #
        # output_r = tf.scatter_mul(output_r, output_indices, credit_a_update)
        # output_g = tf.scatter_mul(output_g, output_indices, credit_a_update)
        # output_b = tf.scatter_mul(output_b, output_indices, credit_a_update)
        #
        # output_r = tf.scatter_add(output_r, output_indices, credit_r_update)
        # output_g = tf.scatter_add(output_g, output_indices, credit_g_update)
        # output_b = tf.scatter_add(output_b, output_indices, credit_b_update)
        #
        # # output setting
        # output_r = tf.reshape(output_r, [output_height, output_width, 1])
        # output_g = tf.reshape(output_g, [output_height, output_width, 1])
        # output_b = tf.reshape(output_b, [output_height, output_width, 1])
        #
        # # # self.fake_image = inputs + output_r[0][indices[0][0]][0][0]
        # tmp_output = tf.concat([output_r, output_g, output_b], axis=2)
        # self.fake_image = bg_img
        # self.fake_image = tf.add(tf.multiply(bg_img, 0), tmp_output)
        # self.fake_image = bg_img * 0 + tmp_output

    def generate(self):
        return self.fake_image
        # return tf.reshape(self.fake_image,
        #                   [1, int(self.fake_image.get_shape()[0]), int(self.fake_image.get_shape()[1]), 3])
