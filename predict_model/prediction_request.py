import argparse

try:
    import googleapiclient.discovery
except ImportError as e:
    print('Please install googleapiclient using: pip3 install google-api-python-client')
    raise e


def main():
    args = _parse_args()

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}/versions/{}'.format(
        args.project_id, args.model_name, args.version_name)

    # This has to be non-empty otherwise google client api throws an error
    dummy_instances = [1, 2, 3]

    response = service.projects().predict(
        name=name,
        body={
            'instances': dummy_instances,
            'prefix': args.prefix,
            'temperature': args.temperature,
            'max_gen_length': args.max_gen_length
        }
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    else:
        print(response['predictions'])


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Script for running prediction from a deployed BlaioPredictor '
                    'using google api client.')
    parser.add_argument(
        '--project-id', type=str, required=True,
        help='Google cloud project id.')
    parser.add_argument(
        '--model-name', type=str, required=True,
        help='The model name. E.g. created with: gcloud ai-platform models create <model-name>.')
    parser.add_argument(
        '--version-name', type=str, required=True,
        help='The name of the version of the model to run. E.g. created with: '
             'gcloud beta ai-platform versions create <version-name>.')
    parser.add_argument(
        '--prefix', type=str, default='Hej bloggen.',
        help='prefix argument to the BlaioPredictor.')
    parser.add_argument(
        '--temperature', type=int, default=0.2,
        help='temperature argument to the BlaioPredictor.')
    parser.add_argument(
        '--max-gen-length', type=int, default=1000,
        help='max_gen_length argument to the BlaioPredictor.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
