import fiftyone as fo
import fiftyone.zoo as foz
import argparse
import os
from csv_od_gcs_exporter import CSVObjectDetectorGoogleStorageBucketExporter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=True,
        help="Fiftyone dataset (https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html).",
    )
    parser.add_argument(
        "--gcs_bucket",
        dest="gcs_bucket",
        required=True,
        help="Google Storage bucket name",
    )
    parser.add_argument(
        "--gcs_project_id",
        dest="gcs_project_id",
        required=True,
        help="Google storage project id",
    )
    parser.add_argument(
        "--classes",
        dest="classes",
        required=True,
        help="Comma separated list of Fiftyone classes (e.g. Cat,Mouse,Hamster)"
    )
    parser.add_argument(
        "--max_samples",
        dest="max_samples",
        required=True,
        help="maximum number of samples"
    )
    parser.add_argument(
        "--tvt_distribution",
        dest="tvt_distribution",
        required=True,
        help="comma separated 'train validate test' distributuon of data set (e.g. 0.7,0.2,0.1)"
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        default=10,
        help="Number of threads for uploading to Google Cloud Storage"
    )
    known_args, pipeline_args = parser.parse_known_args()

    dataset_name = known_args.dataset
    threads = known_args.threads
    max_samples = known_args.max_samples
    gcs_bucket = known_args.gcs_bucket
    gcs_project_id = known_args.gcs_project_id
    classes = known_args.classes.split(",")

    # additional validation
    tvt = known_args.tvt_distribution.split(",")
    if len(tvt) != 3:
        print("tvt requires exactly 3 parameters (e.g. 0.7,0.2,0.1) in order: train,validation,test")
        os.exit(1)
    if len(classes) <= 0:
        print("must have at least 1 class. Check your dataset for available list of classes")
        os.exit(1)

    train_samples = float(tvt[0])
    validation_samples = float(tvt[1])
    test_samples = float(tvt[2])

    dataset = foz.load_zoo_dataset(
        dataset_name, 
        label_types= "detections",
        splits=["train", "validation", "test"], 
        classes=classes,
        max_samples=int(max_samples)
    )

    view = dataset.view()
    
    total_samples = len(view)

    exporter = CSVObjectDetectorGoogleStorageBucketExporter(classes=classes, 
        dataset=dataset_name, 
        gcs_bucket=gcs_bucket, 
        gcs_project_id=gcs_project_id, 
        train_percentage=train_samples, 
        validation_percentage=validation_samples, 
        test_percentage=test_samples,
        total_samples=total_samples
        )
    dataset.export(dataset_exporter=exporter)    

    # session = fo.launch_app(dataset)
    # session.wait()