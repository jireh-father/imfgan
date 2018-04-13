import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

sess = tf.InteractiveSession()

img1 = np.random.random([400, 600, 3])
img2 = np.random.random([400, 600, 3])
img3 = np.random.random([400, 600, 3])
images = np.array([img1, img2, img3])
# images = tf.convert_to_tensor(images)  # it can be uncommented.

img1_crop = [100, 100, 100, 100]
img2_crop = [200, 150, 100, 100]
img3_crop = [150, 200, 100, 100]
crop_values = np.array([img1_crop, img2_crop, img3_crop])


# crop_values = tf.convert_to_tensor(crop_values)  # it can be uncommented.

def crop_image(img, crop):
    return tf.image.crop_to_bounding_box(img,
                                         crop[0],
                                         crop[1],
                                         crop[2],
                                         crop[3])


fn = lambda x: crop_image(x[0], x[1])
elems = (images, crop_values)
print(crop_values)

cropped_image = tf.map_fn(fn, elems=elems, dtype=tf.float64)
result = sess.run(cropped_image)

print(result.shape)

# plt.imshow(result[0])
# plt.show()