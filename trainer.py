from dataset import Dataset
from network import Network
import tensorflow as tf


class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset = Dataset(self.config)
        self.network = Network(config.batch_size, config.input_height, config.input_width, config.real_height,
                               config.real_width, config.learning_rate, config.optimizer, config.optimizer_param)

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

    def train(self):
        try:
            print("Train!!")
            for itr in range(1, self.config.iterations):
                inputs, real_images = self.dataset.batch()
                self.network.train(self.sess, inputs, real_images)

                if itr % 10 == 0:
                    g_loss_val, d_loss_val, summary_str = self.sess.run(
                        [self.gen_loss, self.discriminator_loss, self.summary_op], feed_dict=feed_dict)
                    print("Step: %d, generator loss: %g, discriminator_loss: %g" % (itr, g_loss_val, d_loss_val))
                    self.summary_writer.add_summary(summary_str, itr)

                if itr % 2000 == 0:
                    self.saver.save(self.sess, self.logs_dir + "model.ckpt", global_step=itr)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print("Ending Training...")
        finally:
            self.coord.request_stop()
            self.coord.join(self.threads)  # Wait for threads to finish.
