from generator import Generator
from discriminator import Discriminator
import tensorflow as tf


class Network:
    def __init__(self, input_height, input_width, real_height, real_width, learning_rate, optimizer,
                 optimizer_param, is_training, w_loss=True, transform_mode=1):
        self.input_ph = tf.placeholder(tf.float32, [1, input_height, input_width, 11])
        self.real_image_ph = tf.placeholder(tf.float32, [1, real_height, real_width, 3])
        self.learning_rate = learning_rate

        self.generator = Generator(self.input_ph, is_training, input_height, input_width, real_height, real_width,
                                   mode=transform_mode)
        self.generated_images = self.generator.generate()
        tf.summary.image("generated_image", self.generated_images, max_outputs=10)

        self.discriminator = Discriminator(is_training)

        self.fake_logits = self.discriminator.inference(self.generated_images, False)
        tf.summary.histogram("fake_logits", self.fake_logits)
        if not is_training:
            return

        self.real_logits = self.discriminator.inference(self.real_image_ph)
        tf.summary.histogram("real_logits", self.real_logits)
        # self.fake_logits = self.discriminator.inference(self.real_image_ph)

        if w_loss:
            self._wloss()
        else:
            self._normal_loss()

        train_variables = tf.trainable_variables()
        for v in train_variables:
            Network._add_to_regularization_and_summary(var=v)

        self.generator_variables = [v for v in train_variables if v.name.startswith("generator")]
        # print(map(lambda x: x.op.name, generator_variables))
        self.discriminator_variables = [v for v in train_variables if v.name.startswith("discriminator")]
        # print(map(lambda x: x.op.name, discriminator_variables))

        self.optimizer = self._get_optimizer(optimizer, optimizer_param)

        self.generator_train_op = self._train(self.generator_loss, self.generator_variables, self.optimizer)
        self.discriminator_train_op = self._train(self.discriminator_loss, self.discriminator_variables, self.optimizer)

    def eval(self, sess, inputs):
        return sess.run([self.generated_images,
                         self.generator.bg_transform,
                         self.generator.title_transform,
                         self.generator.credit_transform], feed_dict={self.input_ph: inputs})

    def train(self, sess, inputs, real_images):
        sess.run(self.discriminator_train_op, feed_dict={self.input_ph: inputs, self.real_image_ph: real_images})
        sess.run(self.generator_train_op, feed_dict={self.input_ph: inputs})

    def inference(self, sess, inputs):
        pass

    def _wloss(self):
        self.discriminator_loss = tf.reduce_mean(self.real_logits - self.fake_logits)
        self.generator_loss = tf.reduce_mean(self.fake_logits)

        tf.summary.scalar("Discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("Generator_loss", self.generator_loss)

    def _cross_entropy_loss(self, logits, labels, name="x_entropy"):
        xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar(name, xentropy)
        return xentropy

    def _normal_loss(self):
        discriminator_loss_real = self._cross_entropy_loss(self.real_logits, tf.ones_like(self.real_logits),
                                                           name="disc_real_loss")

        discriminator_loss_fake = self._cross_entropy_loss(self.fake_logits, tf.zeros_like(self.fake_logits),
                                                           name="disc_fake_loss")
        self.discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        gen_loss_disc = self._cross_entropy_loss(self.fake_logits, tf.ones_like(self.fake_logits), name="gen_disc_loss")
        # if use_features:
        #     gen_loss_features = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.crop_image_size ** 2)
        # else:
        #     gen_loss_features = 0
        # self.gen_loss = gen_loss_disc + 0.1 * gen_loss_features
        self.generator_loss = gen_loss_disc
        tf.summary.scalar("Discriminator_loss", self.discriminator_loss)
        tf.summary.scalar("Generator_loss", self.generator_loss)

    @staticmethod
    def _add_to_regularization_and_summary(var):
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))

    def _get_optimizer(self, optimizer_name, optimizer_param):
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(self.learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(self.learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _train(self, loss_val, var_list, optimizer):
        grads = self.optimizer.compute_gradients(loss_val, var_list=var_list)
        print(grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)

        return optimizer.apply_gradients(grads)
