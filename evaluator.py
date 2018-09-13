from dataset import Dataset
from network import Network
import tensorflow as tf
from env import Env
import numpy as np
from PIL import Image
import os


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.dataset = Dataset(config)
        self.env = Env()
        self.network = Network(config.input_height, config.input_width, config.real_height, config.real_width,
                               None, None, None, False, transform_mode=config.transform_mode)

        self._init_session()

    def _init_session(self):
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

        self.sess.run(tf.initialize_all_variables())
        if self.config.checkpoint_dir:
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model restored")

    def _summary(self, step, inputs, real_images):
        summary_str, lo, fl, theta = self.sess.run([self.summary_op, self.network.generator.localization,
                                                    self.network.fake_logits, self.network.generator.theta],
                                                   feed_dict={self.network.input_ph: inputs,
                                                              self.network.real_image_ph: real_images})
        print("Step: %d" % step)
        print("localization", lo)
        print("fake_logits", fl)
        print("theta", theta)
        self.summary_writer.add_summary(summary_str, step)

    def eval(self):
        print("eval!!")
        test_img_path = os.path.join(self.config.log_dir, "test")
        if not os.path.isdir(test_img_path):
            os.makedirs(test_img_path)

        for i in range(1, self.config.eval_cnt):
            inputs, real_images = self.dataset.batch()

            # self._sample()
            result = self.network.eval(self.sess, inputs)
            self._summary(i, inputs, real_images)
            # print(result[0])
            # sys.exit()
            # ii = (result[0][0] - result[0][0].min()) / (result[0][0].max() - result[0][0].min()) * 255
            # img = Image.fromarray(ii, "RGB")
            # img.save(os.path.join(test_img_path, "%d_%s.jpg" % (i, "gen")))
            # ii = (result[1][0] - result[1][0].min()) / (result[1][0].max() - result[1][0].min()) * 255
            # img = Image.fromarray(ii, "RGB")
            # img.save(os.path.join(test_img_path, "%d_%s.jpg" % (i, "bg")))
            #
            # ii = (result[2][0] - result[2][0].min()) / (result[2][0].max() - result[2][0].min()) * 255
            # img = Image.fromarray(ii, "RGBA")
            # img.save(os.path.join(test_img_path, "%d_%s.png" % (i, "title")))
            #
            # ii = (result[2][0] - result[2][0].min()) / (result[2][0].max() - result[2][0].min()) * 255
            # img = Image.fromarray(ii, "RGBA")
            # img.save(os.path.join(test_img_path, "%d_%s.png" % (i, "credit")))
            # ori = np.split(inputs, [3, 7, 11], 3)
            # img = Image.fromarray(ori[0][0], "RGB")
            # img.save(os.path.join(test_img_path, "%d_%s.jpg" % (i, "bg_ori")))
            # img = Image.fromarray(ori[1][0], "RGBA")
            # img.save(os.path.join(test_img_path, "%d_%s.png" % (i, "title_ori")))
            # img = Image.fromarray(ori[2][0], "RGBA")
            # img.save(os.path.join(test_img_path, "%d_%s.png" % (i, "credit_ori")))

    def _sample(self):
        action_probs = self.network.inference()
        return np.argmax(action_probs)
