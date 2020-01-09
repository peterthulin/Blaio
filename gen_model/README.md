# Model training

## Setup
Configure local environment by running `./setupenv.sh` or `pip3 install -r requirements.txt`.  
This assumes you have Keras setup with Tensorflow backend. See official Keras documentation to see how:  
https://keras.io/backend/  
  
*Note:* setup.py is mainly used for maintaining Google Cloud support.

## Cloud training
1. Configure google cloud SDK by following the instructions here: https://cloud.google.com/sdk/docs/  

2. Set the project for google cloud:
`gcloud config set project $PROJECT_ID`  

3. Create the bucket (only needs to be done first time):  
```
gsutil mb -l europe-west1 gs://$BUCKET_NAME
gsutil ls -al gs://$BUCKET_NAME  # Verify access to bucket
```  

4. Submit a training job:  
```
gcloud ai-platform jobs submit training blaio_training \
  --package-path gen_model/ \
  --module-name gen_model.train_model \
  --region $REGION \
  --python-version 3.7 \
  --runtime-version 1.15 \
  --job-dir $BUCKET_NAME/path/to/jobdir \
  --stream-logs -- \
  --blog-data gs://$BUCKET_NAME/path/to/dataset.json
```  

## Local training
```
python3 gen_model/train_model.py --model-name blaio --job-dir somewhere/local \
  --blog-data path/to/dataset.json
```
