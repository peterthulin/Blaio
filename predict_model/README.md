# Running prediction in google cloud
Below follows some sparse instructions for how to configure prediction with google cloud. See the official documentation for more details:  
https://cloud.google.com/ml-engine/docs/tensorflow/custom-prediction-routine-keras  

Distribute predict_model package as tar.gz file:  
`python setup.py sdist --formats=gztar`  

This file needs to be copied to a gcloud bucket location:  
`gsutil cp ./dist/my_custom_code-0.1.tar.gz gs://$BUCKET_NAME/blaio_prediction-0.1.tar.gz`  

Create a model:  
`gcloud ai-platform models create $MODEL_NAME --regions $REGION`  

Install gcloud beta (run once):  
`gcloud components install beta`  

Create a model version to run predictions on:  
```
gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin gs://$BUCKET_NAME/trained_model_location \
  --package-uris gs://$BUCKET_NAME/blaio_prediction-0.1.tar.gz \
  --prediction-class predict_model.BlaioPredictor
```
Installed required libraries for sendin prediction requests:  
`pip3 install --upgrade google-api-python-client`  

Running prediction requests requires authentication:  
https://cloud.google.com/ml-engine/docs/tensorflow/custom-prediction-routine-keras#authenticate_your_gcp_account  

Use prediction_request.py to see how to run create a prediction request:  
`python3 prediction_request.py --help`  
