import argparse
import json
import random

from pathlib import Path

from textgenrnn import textgenrnn


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _SCRIPT_DIR / 'training_config.json'


def main():
    args = _parse_args()

    json_blog_data = read_json_file(args.blog_data)
    if args.dataset_size:
        json_blog_data = sample_json_blog_data(json_blog_data, args.dataset_size)

    texts = convert_json_data_to_texts(json_blog_data)

    training_config = read_json_file(args.config)

    textgenrnn_model = textgenrnn(name=args.model_name)
    textgenrnn_model.reset()
    textgenrnn_model.train_on_texts(texts,
                                    **training_config)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train blog text generation model, aka a Blaio model.')
    parser.add_argument(
        '--blog-data', type=Path, required=True,
        help='Path to json file containing scraped blog data.')
    parser.add_argument(
        '--dataset-size', type=int, default=None,
        help='Use to select a smaller portion of the blog data.')
    parser.add_argument(
        '--model-name', type=str, default='blaigo_model',
        help='Select a model name to alter name of output files.')
    parser.add_argument(
        '--config', type=Path, default=_DEFAULT_CONFIG_PATH,
        help='Config file for textgenrnn model training.')
    return parser.parse_args()


def read_json_file(json_file_path):
    # Reads the blog content json file into list of dictionaries structure
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    return json_data


def sample_json_blog_data(json_blog_data, dataset_size):
    random.seed(1337)  # TODO: parameterize this?
    return random.sample(json_blog_data, dataset_size)


def convert_json_data_to_texts(json_blog_data):
    texts = []
    for blog_dict in json_blog_data:
        title = blog_dict['title']
        content = blog_dict['content']
        text = f"{title}. {content}"
        texts.append(text)
    return texts


if __name__ == '__main__':
    main()
