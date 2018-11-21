#!/usr/bin/env python

import tensorflow as tf

from utils.argparser import parse_args
from utils.dirs import create_dirs

from utils.logger import Logger
from models.lstm import VanillaLSTM as Model
from utils.data_loader import Reader
from actions.train import Trainer
from actions.predict import Predictor
import utils.datapreparation as dataprep


def main():
    args = parse_args()
    create_dirs([args.checkpoint_dir, args.log_dir])

    sess = tf.Session()

    logger = Logger(sess, args)
    model = Model(args, logger)


    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("testing/graph", sess.graph)
    writer.close()


    #dataset = (dataprep.load_all_dataset("data/training/piano_roll_fs5/"))
    #dataprep.embed_play_v1(dataset[0])
    #dataprep.piano_roll_to_mid_file(dataset[0]*100, "test")


    reader = Reader(args, sess, logger)

    if args.action == 'train':
        trainer = Trainer(sess, model, reader, args, logger)
        trainer.train()
    else:
        predictor = Predictor(sess, model, reader, args, logger)
        predictor.predict()


main()
