import argparse
import glob
import json
import os
import random
import shutil

from pathlib import Path

from textgenrnn import textgenrnn

from tensorflow.io import gfile


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = {
    "batch_size": 128,
    "dim_embeddings": 300,
    "dropout": 0.2,
    "gen_epochs": 1,
    "max_gen_length": 300,
    "max_length": 50,
    "multi_gpu": False,
    "new_model": True,
    "num_epochs": 3,
    "rnn_bidirectional": False,
    "rnn_size": 64,
    "save_epochs": 1,
    "verbose": 1
}


def main():
    args = _parse_args()

    if is_gscloud_path(args.blog_data):
        json_blog_data = read_gscloud_file(args.blog_data)
    else:
        json_blog_data = read_json_file(args.blog_data)

    if args.dataset_size:
        json_blog_data = sample_json_blog_data(json_blog_data, args.dataset_size)

    if args.job_dir:
        if is_gscloud_path(args.job_dir):
            gfile.makedirs(args.job_dir)
        else:
            Path(args.job_dir).mkdir(exist_ok=True, parents=True)

    texts = convert_json_data_to_texts(json_blog_data)

    training_config = _DEFAULT_CONFIG
    if args.config:
        training_config.update(read_json_file(args.config))

    textgenrnn_model = textgenrnn(name=args.model_name)
    textgenrnn_model.reset()
    textgenrnn_model.train_on_texts(texts,
                                    **training_config)

    if args.job_dir:
        if is_gscloud_path(args.job_dir):
            move_output_files_to_cloud(args.model_name, args.job_dir)
        else:
            move_output_files_to_dir(args.model_name, args.job_dir)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Train blog text generation model, aka a Blaio model.')
    parser.add_argument(
        '--blog-data', type=str, required=True,
        help='Path to json file containing scraped blog data.')
    parser.add_argument(
        '--dataset-size', type=int, default=None,
        help='Use to select a smaller portion of the blog data.')
    parser.add_argument(
        '--model-name', type=str, default='blaigo_model',
        help='Select a model name to alter name of output files.')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Config file for textgenrnn model training.')
    parser.add_argument(
        '--job-dir', type=str, default=None,
        help='Path to move all output files to.')
    return parser.parse_args()


def read_json_file(json_file_path):
    # Reads the blog content json file into list of dictionaries structure
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
    return json_data


def is_gscloud_path(file_path):
    return file_path.startswith('gs:')


def read_gscloud_file(gs_uri):
    raw_content = gfile.GFile(gs_uri).read()
    json_data = json.loads(raw_content)
    return json_data


def create_gscloud_dir(gs_uri):
    gfile.makedirs(gs_uri)


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


def move_output_files_to_dir(model_name: str, out_dir: str):
    for f in glob.glob('./' + model_name + '*'):
        shutil.move(f, out_dir)


def move_output_files_to_cloud(model_name: str, cloud_dir: str):
    for f in glob.glob('./' + model_name + '*'):
        filename = Path(f).name
        gfile.copy(f, cloud_dir + '/' + filename)
        os.remove(f)


if __name__ == '__main__':
    main()
