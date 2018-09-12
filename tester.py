import tensorflow as tf
from evaluator import Evaluator

tf.flags.DEFINE_string("poster_dir", "F:\data\imfgan/poster", "path to poster directory")
tf.flags.DEFINE_string("bg_dir", "F:\data\imfgan/bg", "path to bg directory")
tf.flags.DEFINE_string("title_dir", "F:\data\imfgan/title", "path to title directory")
tf.flags.DEFINE_string("credit_dir", "F:\data\imfgan/credit", "path to credit directory")
tf.flags.DEFINE_integer("input_width", 250, "input_width")
tf.flags.DEFINE_integer("input_height", 300, "input_height")
tf.flags.DEFINE_integer("real_width", 150, "real_width")
tf.flags.DEFINE_integer("real_height", 200, "real_height")
tf.flags.DEFINE_integer("eval_cnt", 100, "eval count")
tf.flags.DEFINE_integer("batch_size", 1, "F:\develop\imfgan\main.py")
tf.flags.DEFINE_string("log_dir", "eval_log", "path to logs directory")
tf.flags.DEFINE_boolean("dataset_preload", False, "dataset memory load")
tf.flags.DEFINE_string("checkpoint_dir", "cp", "checkpoint dir")

FLAGS = tf.flags.FLAGS


def main(argv=None):
    evaluator = Evaluator(FLAGS)
    evaluator.eval()


if __name__ == "__main__":
    tf.app.run()
