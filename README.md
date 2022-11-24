# FiftyOne Zoo Labeled Image Dataset Exporter to Google's AutoML Vision Object Detection

To use the importData method with tflite, both the CSV file and the images it points to must be in a Google Cloud Storage bucket.

Example:
```
 TRAIN,gs://folder/image1.png,car,0.1,0.1,,,0.3,0.3,,
 TRAIN,gs://folder/image1.png,bike,.7,.6,,,.8,.9,,
 UNASSIGNED,gs://folder/im2.png,car,0.1,0.1,0.2,0.1,0.2,0.3,0.1,0.3
 TEST,gs://folder/im3.png,,,,,,,,,
```

> *Warning*
> This script exports images to google storage cloud therefor some costs my infer. Fiftyone is working with public datasets so this scripts uploads images and makes them public. 

![FyftyOne Exporter for Google AutoML](assets/Screenshot%20from%202022-11-18%2014-02-06.png "FyftyOne custom data exporter")

## Prerequsite

Login to your google account and authenticate SDK
```
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUPROJECTID
```

> **Warning**
> Many times `rm -rf ~/.fifytone/` needs to be called from cmd due to Fifytone error `Subprocess ['/home/igor/workspace/coral/.venv/lib/python3.10/site-packages/fiftyone/db/bin/mongod', '--dbpath', '/home/igor/.fiftyone/var/lib/mongo', '--logpath', '/home/igor/.fiftyone/var/lib/mongo/log/mongo.log', '--port', '0', '--nounixsocket'] exited with error 100:`

[Official issue](https://github.com/voxel51/fiftyone/issues/845)

Don't forget to create a bucket in your projects storage. 

## Install

Create virtual env and install requirements:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
python3 main.py --dataset open-image-v6 --gcs_bucket mybucket --gcs_project_id mygoogleprojectId --classes Mouse,Hamster,Cat --max_samples 300 --tvt_distribution 0.7,0.2,0.1
```

Parameters explained:
- `dataset` - dataset name One of [Fiftyone datasets](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html)
- `gcs_bucket` - the name of the Google Clout Storage bucket
- `gcs_project_id` - the name of the Goole Project Id where the gcs_bucket was created
- `classes` - Based on the dataset, the classes to transform
- `max_samples` - a maximum number of samples to load per split. Splits are hard coded in this code as (["train", "validation", "test"])
- `tvt_distribution` - perfenctage of specific class to be marked as TRAIN, VALIDATE or TEST in the output CSV file. Example: 0.7,0.2,0.1 means 70% for training, 20% for validation and 10% for testing. The tvt distribution in per class.

The output is `labels.csv` file uploaded to your defined storage bucket. 