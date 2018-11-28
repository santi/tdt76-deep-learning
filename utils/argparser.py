import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='specify batch size')

    parser.add_argument('--epochs',
                        type=int,
                        default=10000,
                        help='number of epochs to run')

    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints/',
                        help='model checkpoint directory')

    parser.add_argument('--log_dir',
                        type=str,
                        default='log/',
                        help='log summary directory')

    parser.add_argument('--output_dir',
                        type=str,
                        default='output/',
                        help='output directory')

    parser.add_argument('--model_name',
                        type=str,
                        default='model',
                        help='model name')

    parser.add_argument('--action',
                        type=str,
                        default='train',
                        choices=['train', 'predict', 'train_composer'])

    parser.add_argument('--composer',
                        type=str,
                        choices=['mz', 'bach', 'brahms', 'debussy'])

    parser.add_argument('--training_data',
                        type=str,
                        default='data/training/piano_roll_fs1/')

    parser.add_argument('--validation_data',
                        type=str,
                        default='data/validation/piano_roll_fs1/')

    parser.add_argument('--prediction_data',
                        type=str,
                        default='data/predicting/piano_roll_fs1/')

    parser.add_argument('--debug',
                        type=bool,
                        default=True)
    

    return parser.parse_args()
