import tensorflow as tf


class Discriminator:
    def __init__(self, is_training):
        self.is_training = is_training

    def inference(self, inputs, scope_reuse=True):
        def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
            """
            Code taken from http://stackoverflow.com/a/34634291/2267819
            """
            with tf.variable_scope(scope):
                beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                                       , trainable=True)
                gamma = tf.get_variable(name='gamma', shape=[n_out],
                                        initializer=tf.random_normal_initializer(1.0, stddev),
                                        trainable=True)
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=decay)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(phase_train,
                                    mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
            return normed

        def leaky_relu(x):
            return tf.maximum(0.2 * x, x)

        def kernel_initializer():
            return tf.truncated_normal_initializer(0.0, 0.02)

        def bias_initializer():
            return tf.constant_initializer(0.0)

        with tf.variable_scope("discriminator") as scope:
            if scope_reuse:
                scope.reuse_variables()
            print(inputs)
            net = tf.layers.conv2d(inputs, 64, 4, strides=(2, 2), padding="SAME", activation=leaky_relu, name="d_conv1",
                                   kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer())

            net = tf.layers.conv2d(net, 128, 4, strides=(2, 2), padding="SAME", name="d_conv2",
                                   kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer())
            # net = batch_norm(net, 128, self.is_training, scope="disc_bn1")
            net = leaky_relu(net)

            net = tf.layers.conv2d(net, 256, 4, strides=(2, 2), padding="SAME", name="d_conv3",
                                   kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer())
            # net = batch_norm(net, 256, self.is_training, scope="disc_bn2")
            net = leaky_relu(net)

            net = tf.layers.conv2d(net, 512, 4, strides=(2, 2), padding="SAME", name="d_conv4",
                                   kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer())
            # net = batch_norm(net, 512, self.is_training, scope="disc_bn3")
            net = leaky_relu(net)

            net = tf.layers.conv2d(net, 1, 4, strides=(2, 2), padding="SAME", name="d_conv5",
                                   kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer())

            return net
