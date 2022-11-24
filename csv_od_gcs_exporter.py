import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.data as foud

import os
import csv

import concurrent
import hashlib
import numpy as np

from google.cloud import storage
from PIL import Image
import io

from concurrent.futures import ThreadPoolExecutor

tp = ThreadPoolExecutor(10)  # max 10 threads


def call_with_future(fn, future, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)

def threaded(fn):
    def wrapper(*args, **kwargs):
        return tp.submit(fn, *args, **kwargs)  # returns Future object
    return wrapper

class CSVObjectDetectorGoogleStorageBucketExporter(foud.LabeledImageDatasetExporter):
    """
    To use the importData method with tflite, both the CSV file and the images it points to must be in a Google Cloud Storage bucket.

    Additionally, the CSV file must also fulfill the following requirements:

    - The file can have any filename, but must be in the same bucket as your image file.
    - Must be UTF-8 encoded.
    - Must end with a .csv extension.
    - Has one row for each bounding box in the set you are uploading, or one row for each image with no bounding box (such as row 4 below).
    - Contain one image per line; an image with multiple bounding boxes will be repeated on as many rows as there are bounding boxes.

    Example rows:
    TRAIN,gs://folder/image1.png,car,0.1,0.1,,,0.3,0.3,,
    TRAIN,gs://folder/image1.png,bike,.7,.6,,,.8,.9,,
    UNASSIGNED,gs://folder/im2.png,car,0.1,0.1,0.2,0.1,0.2,0.3,0.1,0.3
    TEST,gs://folder/im3.png,,,,,,,,,

    """
    def __init__(self, **kwargs):
        super().__init__(export_dir=None)
        self._labels = None
        self._upload_futures = None
        self.sclient = None
        self.sbucket = kwargs.get("gcs_bucket")        
        self.dataset = kwargs.get("dataset")
        self.projectId = kwargs.get("gcs_project_id")
        self.classes = kwargs.get("classes")
        self.train_percent = kwargs.get("train_percentage")
        self.validation_percent = kwargs.get("validation_percentage")
        self.test_percent = kwargs.get("test_percentage")
        self.total_samples = kwargs.get("total_samples")

        self.ds_distribution_map = {} # map where counting of how many traning, test or validation samples each category has. init with 0
        for cls in self.classes:
            self.ds_distribution_map[cls] = {
                "TRAIN": 0,
                "TEST": 0,
                "VALIDATE": 0,
            }

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) exported by this
        exporter.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            exporter directly exports labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the exporter can export a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the exporter can handle label dictionaries with
            value-types specified by this dictionary. Not all keys need be
            present in the exported label dicts
        -   ``None``. In this case, the exporter makes no guarantees about the
            labels that it can export
        """
        #todo: add to input
        return fo.Detections
    
    @property
    def requires_image_metadata(self):
        """Whether this exporter requires
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each sample
        being exported.
        """
        # Return True or False here
        return False

    def setup(self):
        """Performs any necessary setup before exporting the first sample in
        the dataset.

        This method is called when the exporter's context manager interface is
        entered, :func:`DatasetExporter.__enter__`.
        """
        self._labels = []

        self._upload_futures = []
        self.sclient = storage.Client(project=self.projectId)
        self.sbucket = self.sclient.get_bucket(self.sbucket)
    
    @threaded
    def upload_file(self, image_or_path, upload_img_name):
        blob = self.sbucket.blob(self.dataset + "/" + upload_img_name)
        if blob.exists():
            return f"gs://{self.sbucket}/{self.dataset}/{upload_img_name}"

        if type(image_or_path) is np.ndarray:
            img = Image.fromarray(image_or_path)
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format = "jpeg")
            blob.upload_from_string(imgByteArr.getvalue(), content_type='image/jpeg')
        else:
            blob.upload_from_filename(image_or_path)

        blob.make_public()
        return f"gs://{self.sbucket}/{self.dataset}/{upload_img_name}"

    def export_sample(self, image_or_path, label, metadata=None):
        """Exports the given sample to the dataset.

        Args:
            image_or_path: an image or the path to the image on disk
            metadata (None): a :class:`fiftyone.core.metadata.ImageMetadata`
                isinstance for the sample. Only required when
                :meth:`requires_image_metadata` is ``True``
        """
        # out_image_path, _ = self._image_exporter.export(image_or_path)
        if metadata is None:
            metadata = fo.ImageMetadata.build_for(image_or_path)

        # Name of the object to be stored in the bucket
        img_name = None
        if type(image_or_path) is np.ndarray:
            img_name = hashlib.md5(image_or_path.tobytes(order='C')).hexdigest() + ".jpg"
        else:
            img_name = image_or_path.split("/")[-1]

        # upload file to GCS bucket
        future = self.upload_file(image_or_path, img_name)
        self._upload_futures.append(future)

        items = []
        for detection in label.detections:
            if "bounding_box" in detection and "label" in detection:
                if detection["label"] in self.classes:
                    lbl = detection.label
                    # increase counts and determine dataset (train,validate,test)
                    determined_dataset = "TEST"
                    if self.ds_distribution_map[lbl]["TEST"] >= self.test_percent * self.total_samples:
                        if self.ds_distribution_map[lbl]["VALIDATE"] >= self.validation_percent * self.total_samples:
                            determined_dataset = "TRAIN"
                        else:
                            self.ds_distribution_map[lbl]["VALIDATE"] += 1
                            determined_dataset = "VALIDATE"
                    else:
                        self.ds_distribution_map[lbl]["TEST"] += 1

                    bBox = detection["bounding_box"]
                    items.append((
                        determined_dataset,
                        img_name,
                        lbl,
                        bBox[0],
                        bBox[1],
                        '',
                        '',
                        bBox[2],
                        bBox[3],
                        '',
                        '',
                    ))

        if len(items) > 0:
            # concat arrays
            self._labels += items

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        exported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetExporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetExporter.__exit__`
        """
        for future in concurrent.futures.as_completed(self._upload_futures):
            print(f"{future.result()} uploaded")

        output = io.StringIO()
        writer = csv.writer(output)
        for row in self._labels:
            writer.writerow(row)

        output_csv = output.getvalue()
        blob = self.sbucket.blob(self.dataset + "/labels.csv")
        blob.upload_from_string(output_csv, content_type='text/csv')
        
        tp.shutdown()