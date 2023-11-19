import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', required=True, type=str, help='Name of the model architecture.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained models.')
    parser.add_argument('--data_name', type=str, help='Name of the dataset.')
    parser.add_argument('--deterministic', type=bool, default=True, help='Whether to be deterministic')

    return parser


def get_params() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_args()
