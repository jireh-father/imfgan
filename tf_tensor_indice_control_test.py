import tensorflow as tf

l = tf.convert_to_tensor([0, 0, 1, 0])
o = tf.convert_to_tensor([52, 32, 44, 22])
hh = tf.Variable([4, 5, 11, 235, 44])
print(l)
e = tf.equal(l, 0)
indices = tf.where(e)
print(indices)
# Pick rows that contribute to the loss and filter out the rest.
rpn_class_logits = tf.gather_nd(o, indices)
indices = tf.reshape(indices, [4])
ad = tf.scatter_update(hh, indices, o)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
a, b, c, d,e  = sess.run([e, indices, rpn_class_logits, ad, hh])
print(a, b)
print(c)
print(d)
print(e)
