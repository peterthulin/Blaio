from setuptools import setup

setup(
    name='blaio_prediction',
    version='0.1',
    scripts=['predict_model.py'],
    install_requires=['textgenrnn==1.5.0']
)
