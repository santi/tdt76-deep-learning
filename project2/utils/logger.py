import os
import sys
import tensorflow as tf

class Logger:
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.summary_writer = None


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
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.args.log_dir), self.sess.graph)
        self.summary_writer.add_summary(summaries, global_step)
        self.summary_writer.flush()
