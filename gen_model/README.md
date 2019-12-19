# Model training

## Setup
Configure environment by running `./setupenv.sh` or `pip3 install -r requirements.txt`.  
This assumes you have Keras setup with Tensorflow backend. See official Keras documentation to see how:  
https://keras.io/backend/

## train_model.py
Call train_model.py as such:  
`python3 train_model.py --blog-data /path/to/blog-content.json`  
Use `--help` to see configuration choices.