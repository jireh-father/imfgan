import tensorflow as tf
import numpy as np
from PIL import Image
import sys


def open_image(image_path):
    return Image.open(image_path)


def resize(img, size):
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img)


input_width = 450
input_height = 600
output_width = 400
output_height = 500

# min_color_val = -1.
min_color_val = 0.

i1 = resize(open_image("f:\\1.jpg"), (input_width, input_height))
i2 = resize(open_image("f:\\2.png"), (input_width, input_height))
i3 = resize(open_image("f:\\3.png"), (input_width, input_height))
image_input = np.concatenate((i1, i2, i3), axis=2)

input = tf.placeholder(tf.float32, [1, input_height, input_width, 11])

manifold = tf.Variable(tf.random_uniform([10]))
output_r = tf.Variable(tf.zeros([output_width * output_height], tf.float32))
output_g = tf.Variable(tf.zeros([output_width * output_height], tf.float32))
output_b = tf.Variable(tf.zeros([output_width * output_height], tf.float32))

split0, split1, split2 = tf.split(input, [3, 4, 4], 3)
# print(split0, split1, split2)

split0 = tf.squeeze(split0)
split1 = tf.squeeze(split1)
split2 = tf.squeeze(split2)

# handle bg manifold
bg_manifold = tf.gather(manifold, [0, 1])
bg_x = tf.cast(bg_manifold[0] * (input_width - output_width), tf.int32)
bg_y = tf.cast(bg_manifold[1] * (input_height - output_height), tf.int32)

bg_img = tf.image.crop_to_bounding_box(split0, bg_y, bg_x, output_height, output_width)

# handle title manifold
credit_manifold = tf.gather(manifold, [2, 3, 4, 5])
title_w = tf.cast(credit_manifold[2] * (output_width - 1) + 1, tf.int32)
title_h = tf.cast(credit_manifold[3] * (output_height - 1) + 1, tf.int32)
title_x = tf.cast(credit_manifold[0] * output_width, tf.int32)
title_y = tf.cast(credit_manifold[1] * output_height, tf.int32)
split1 = tf.image.resize_images(split1, (title_h, title_w))
title_crop_w = title_w - tf.nn.relu(title_x + title_w - output_width)
title_crop_h = title_h - tf.nn.relu(title_y + title_h - output_height)
title_img = tf.image.crop_to_bounding_box(split1, 0, 0, title_crop_h, title_crop_w)

# handle credit manifold
credit_manifold = tf.gather(manifold, [6, 7, 8, 9])
credit_w = tf.cast(credit_manifold[2] * (output_width - 1) + 1, tf.int32)
credit_h = tf.cast(credit_manifold[3] * (output_height - 1) + 1, tf.int32)
credit_x = tf.cast(credit_manifold[0] * output_width, tf.int32)
credit_y = tf.cast(credit_manifold[1] * output_height, tf.int32)
split2 = tf.image.resize_images(split2, (credit_h, credit_w))
credit_crop_w = credit_w - tf.nn.relu(credit_x + credit_w - output_width)
credit_crop_h = credit_h - tf.nn.relu(credit_y + credit_h - output_height)
credit_img = tf.image.crop_to_bounding_box(split2, 0, 0, credit_crop_h, credit_crop_w)

# split bg by channel
bg_r, bg_g, bg_b = tf.split(bg_img, 3, 2)
bg_r = tf.reshape(bg_r, [-1])
bg_g = tf.reshape(bg_g, [-1])
bg_b = tf.reshape(bg_b, [-1])

output_r = output_r.assign_add(bg_r)
output_g = output_g.assign_add(bg_g)
output_b = output_b.assign_add(bg_b)

# lay title over bg!
title_r, title_g, title_b, title_alpha = tf.split(title_img, 4, 2)

title_alpha = (title_alpha - tf.reduce_min(title_alpha)) / tf.reduce_max(title_alpha) - tf.reduce_min(title_alpha)

title_r = tf.reshape(title_r * title_alpha, [-1])
title_g = tf.reshape(title_g * title_alpha, [-1])
title_b = tf.reshape(title_b * title_alpha, [-1])
title_a = tf.reshape(1 - title_alpha, [-1])

title_alpha_1d = tf.reshape(title_alpha, [-1])
title_tmp_indices = tf.where(tf.not_equal(title_alpha_1d, min_color_val))
title_tmp_indices = tf.cast(title_tmp_indices, dtype=tf.int32)

title_r_update = tf.reshape(tf.gather(title_r, title_tmp_indices), [-1])
title_g_update = tf.reshape(tf.gather(title_g, title_tmp_indices), [-1])
title_b_update = tf.reshape(tf.gather(title_b, title_tmp_indices), [-1])
title_a_update = tf.reshape(tf.gather(title_a, title_tmp_indices), [-1])

title_color_indices = tf.where(tf.not_equal(title_alpha, min_color_val))
title_color_indices = tf.cast(title_color_indices, dtype=tf.int32)
title_color_indices = title_color_indices + tf.Variable([title_y, title_x, 0], dtype=tf.int32)
title_color_indices1, title_color_indices2, title_color_indices3 = tf.split(title_color_indices, 3, axis=1)
output_indices = title_color_indices1 * output_width + title_color_indices2

output_indices = tf.reshape(output_indices, [-1])

output_r = tf.scatter_mul(output_r, output_indices, title_a_update)
output_g = tf.scatter_mul(output_g, output_indices, title_a_update)
output_b = tf.scatter_mul(output_b, output_indices, title_a_update)

output_r = tf.scatter_add(output_r, output_indices, title_r_update)
output_g = tf.scatter_add(output_g, output_indices, title_g_update)
output_b = tf.scatter_add(output_b, output_indices, title_b_update)

# lay credit over bg!
credit_r, credit_g, credit_b, credit_alpha = tf.split(credit_img, 4, 2)

credit_alpha = (credit_alpha - tf.reduce_min(credit_alpha)) / tf.reduce_max(credit_alpha) - tf.reduce_min(credit_alpha)

credit_r = tf.reshape(credit_r * credit_alpha, [-1])
credit_g = tf.reshape(credit_g * credit_alpha, [-1])
credit_b = tf.reshape(credit_b * credit_alpha, [-1])
credit_a = tf.reshape(1 - credit_alpha, [-1])

credit_alpha_1d = tf.reshape(credit_alpha, [-1])
credit_tmp_indices = tf.where(tf.not_equal(credit_alpha_1d, min_color_val))
credit_tmp_indices = tf.cast(credit_tmp_indices, dtype=tf.int32)

credit_r_update = tf.reshape(tf.gather(credit_r, credit_tmp_indices), [-1])
credit_g_update = tf.reshape(tf.gather(credit_g, credit_tmp_indices), [-1])
credit_b_update = tf.reshape(tf.gather(credit_b, credit_tmp_indices), [-1])
credit_a_update = tf.reshape(tf.gather(credit_a, credit_tmp_indices), [-1])

credit_color_indices = tf.where(tf.not_equal(credit_alpha, min_color_val))
credit_color_indices = tf.cast(credit_color_indices, dtype=tf.int32)
credit_color_indices = credit_color_indices + tf.Variable([credit_y, credit_x, 0], dtype=tf.int32)
credit_color_indices1, credit_color_indices2, credit_color_indices3 = tf.split(credit_color_indices, 3, axis=1)
output_indices = credit_color_indices1 * output_width + credit_color_indices2

output_indices = tf.reshape(output_indices, [-1])

output_r = tf.scatter_mul(output_r, output_indices, credit_a_update)
output_g = tf.scatter_mul(output_g, output_indices, credit_a_update)
output_b = tf.scatter_mul(output_b, output_indices, credit_a_update)

output_r = tf.scatter_add(output_r, output_indices, credit_r_update)
output_g = tf.scatter_add(output_g, output_indices, credit_g_update)
output_b = tf.scatter_add(output_b, output_indices, credit_b_update)

# output setting
output_r = tf.reshape(output_r, [output_height, output_width, 1])
output_g = tf.reshape(output_g, [output_height, output_width, 1])
output_b = tf.reshape(output_b, [output_height, output_width, 1])

output = tf.concat([output_r, output_g, output_b], axis=2)
# output = tf.reshape(output, [output_height, output_width, 3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

output_img = sess.run(output, feed_dict={input: np.array([image_input])})
Image.fromarray(output_img.astype(np.uint8)).save("output.jpg")
