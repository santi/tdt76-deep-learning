import os
import sys
import tensorflow as tf

class Logger:
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.summary_writer = None
        self.saver = None


    def log(self, msg):
        if self.args.debug:
            sys.stdout.write('\033[1;34m')
            print(msg)
            sys.stdout.write('\033[0;0m')
            sys.stdout.flush()


    def summarize(self, summaries, global_step):
        """
        :param summaries: tf summaries to be written
        :param global_step: the current global step (integer)
        :return:
        """

        if not self.summary_writer:
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.args.log_dir, self.args.model_name), self.sess.graph)
        self.summary_writer.add_summary(summaries, global_step)
        self.summary_writer.flush()

    def save_checkpoint(self, global_step):
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep=2)
        self.log(f"Saving model...")
        save_path = self.saver.save(self.sess, os.path.join(self.args.checkpoint_dir, self.args.model_name, self.args.model_name), global_step)
        self.log(f"Model {self.args.model_name} saved to {save_path}. Global step: {global_step}")


    def load_checkpoint(self, checkpoint_path):
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep=2)
        self.log(f"Loading model from {checkpoint_path}...")
        self.saver.restore(self.sess, checkpoint_path)
        global_step = self.sess.run(self.sess.graph.global_step_tensor)
        self.log(f"Model loaded. Global step: {global_step}")