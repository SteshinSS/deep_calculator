import argparse
import pathlib

DEFAULT_CHECKPOINT_PATH = pathlib.Path.cwd()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=pathlib.Path,
        help="Path to model's config",
        required=True
    )
    parser.add_argument('--train', action='store_true', help="Retrain model")
    parser.add_argument(
        '--checkpoint',
        type=pathlib.Path,
        help="Path to model's checkpoint",
    )
    parser.add_argument(
        '--epochs', default=100, help="Number of epochs", type=int
    )
    args = parser.parse_args()

    if not args.checkpoint:
        if args.train:
            args.checkpoint = DEFAULT_CHECKPOINT_PATH
        else:
            print("Provide model's checkpoint to run evaluation.")
            exit(1)

    return args
