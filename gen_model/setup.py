from setuptools import setup

setup(
    name='gen_model',
    packages=['gen_model'],
    version='0.0.1',
    description='Train blaio model.',
    author='Peter Thulin',
    author_email='peter.thulin@aith.se',
    classifiers=[],
    python_requires='>=3',
    include_package_data=True,
    install_requires=['tensorflow==1.15.2', 'textgenrnn==1.5.0']
)
