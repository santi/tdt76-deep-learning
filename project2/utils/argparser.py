import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        type=int,
                        default=32, # TODO: 64
                        help='specify batch size')

    parser.add_argument('--epochs',
                        type=int,
                        default=100000,
                        help='number of epochs to run')

    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints/',
                        help='checkpoint directory')

    parser.add_argument('--log_dir',
                        type=str,
                        default='log/',
                        help='directory of logs')

    parser.add_argument('--action',
                        type=str,
                        default='train',
                        choices=['train', 'predict'])

    parser.add_argument('--training_data',
                        type=str,
                        default='data/training_split.csv')

    parser.add_argument('--validation_data',
                        type=str,
                        default='data/valid_TA_split.csv')

    parser.add_argument('--prediction_data',
                        type=str,
                        default='data/test_nlu18_split.csv')

    parser.add_argument('--debug',
                        type=bool,
                        default=True)

    return parser.parse_args()
