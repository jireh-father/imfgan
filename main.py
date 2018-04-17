import tensorflow as tf
from trainer import Trainer

tf.flags.DEFINE_string("poster_dir", "F:\data\imfgan/poster", "path to poster directory")
tf.flags.DEFINE_string("bg_dir", "F:\data\imfgan/bg", "path to bg directory")
tf.flags.DEFINE_string("title_dir", "F:\data\imfgan/title", "path to title directory")
tf.flags.DEFINE_string("credit_dir", "F:\data\imfgan/credit", "path to credit directory")
tf.flags.DEFINE_integer("batch_size", 1, "batch size")
tf.flags.DEFINE_integer("input_width", 300, "input_width")
tf.flags.DEFINE_integer("input_height", 250, "input_height")
tf.flags.DEFINE_integer("real_width", 200, "real_width")
tf.flags.DEFINE_integer("real_height", 150, "real_height")
tf.flags.DEFINE_float("learning_rate", "2e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("optimizer_param", "0.5", "beta1 for Adam optimizer / decay for RMSProp")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer to use for training")
tf.flags.DEFINE_string("log_dir", "logs", "path to logs directory")
tf.flags.DEFINE_string("checkpoint_dir", None, "checkpoint dir")
tf.flags.DEFINE_float("iterations", "1e5", "No. of iterations to train model")
tf.flags.DEFINE_boolean("dataset_preload", True, "dataset memory load")

FLAGS = tf.flags.FLAGS


def main():
    trainer = Trainer(FLAGS)
    trainer.train()


if __name__ == "__main__":
    tf.app.run()
