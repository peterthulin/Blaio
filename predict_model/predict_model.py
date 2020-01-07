import os

from textgenrnn import textgenrnn


class BlaioPredictor(object):
    """Class to make it possible to run predictions in Googles AI platorm."""

    def __init__(self, model):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Args:
            instances: Not used
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        temperature = [kwargs.get('temperature', 0.2)]
        prefix = kwargs.get('prefix', '')
        max_gen_length = kwargs.get('max_gen_length', 1000)

        text_list = self._model.generate(
            n=1,
            return_as_list=True,
            temperature=temperature,
            prefix=prefix,
            max_gen_length=max_gen_length)

        return text_list

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of BlaioPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. BlaioPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the model and should
                contain the following files: model.hdf5, model.json and vocab.json.

        Returns:
            An instance of `BlaioPredictor`.
        """
        weights_path = os.path.join(model_dir, 'weights.hdf5')
        vocab_path = os.path.join(model_dir, 'vocab.json')
        config_path = os.path.join(model_dir, 'config.json')

        textgenrnn_model = textgenrnn(
            weights_path=weights_path,
            vocab_path=vocab_path,
            config_path=config_path,
            name="blaio")

        return cls(textgenrnn_model)
