#!/usr/bin/env python

import tensorflow as tf

from utils.argparser import parse_args
from utils.dirs import create_dirs

from utils.logger import Logger
from models.model import Model
from utils.data_loader import Reader
from actions.train import Trainer
from actions.predict import Predictor


def main():
    args = parse_args()
    create_dirs([args.checkpoint_dir, args.log_dir])

    sess = tf.Session()

    logger = Logger(sess, args)
    model = Model(args, logger)
    reader = Reader(sess, args, logger)

    if args.action == 'train':
        trainer = Trainer(sess, model, reader, args, logger)
        trainer.train()
    else:
        predictor = Predictor(sess, model, reader, args)
        predictor.predict()


if __name__ == '__main__':
    main()
