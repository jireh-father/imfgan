from dataset import Dataset
from network import Network
import tensorflow as tf


class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset = Dataset(self.config)
        self.network = Network(config.batch_size, config.input_height, config.input_width, config.real_height,
                               config.real_width, config.learning_rate, config.optimizer, config.optimizer_param,
                               config.is_training)

        self._init_session()

    def _init_session(self):
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

        self.sess.run(tf.initialize_all_variables())
        if self.config.checkpoint_dir:
            ckpt = tf.train.get_checkpoint_state(self.config.log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model restored")

    def _summary(self, step, inputs, real_images):
        g_loss_val, d_loss_val, summary_str = self.sess.run(
            [self.network.generator_loss, self.network.discriminator_loss, self.summary_op],
            feed_dict={self.network.input_ph: inputs, self.network.real_image_ph: real_images})
        print("Step: %d, generator loss: %g, discriminatorloss: %g" % (step, g_loss_val, d_loss_val))
        self.summary_writer.add_summary(summary_str, step)

    def _save(self, i):
        self.saver.save(self.sess, self.config.log_dir + "model.ckpt", global_step=i)

    def train(self):
        print("Train!!")
        for i in range(1, self.config.iterations):
            inputs, real_images = self.dataset.batch()
            self.network.train(self.sess, inputs, real_images)

            if i % self.config.summary_interval == 0:
                self._summary(i, inputs, real_images)

            if i % self.config.save_interval == 0:
                self._save(i)
