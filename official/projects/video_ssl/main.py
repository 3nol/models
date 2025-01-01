# -- LIBRARY IMPORTS & PREPARTION --

from utils import logger, RESOURCES_DATA, RESOURCES_EVALS
from .prepare import *
from .config import override_for_ucf101

pp = pprint.PrettyPrinter(indent=4)
logger.info(f"Running TensorFlow {tf.__version__}...")

# -- TF PROJECT IMPORTS --

from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
from official.vision.data import tfrecord_lib

from official.projects.video_ssl.modeling import video_ssl_model
from official.projects.video_ssl.tasks import linear_eval
from official.projects.video_ssl.tasks import pretrain
from official.vision import registry_imports
from official.vision.serving import export_saved_model_lib

# -- CODE --


def process_record(record):
    """
    Convert training samples to SequenceExample format. For more detailed
    explaination about SequenceExample, please check here
    https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample

    Args:
      record: training example with image frames and corresponding label.

    Return:
      Return a SequenceExample which represents a
      sequence of features and some context.
    """
    seq_example = tf.train.SequenceExample()
    for example in record[0]:
        seq_example.feature_lists.feature_list.get_or_create(
            "image/encoded"
        ).feature.add().bytes_list.value[:] = [tf.io.encode_jpeg(example).numpy()]
    seq_example.context.feature["clip/label/index"].int64_list.value[:] = [
        record[1].numpy()
    ]
    return seq_example


output_dir = "../ucf101_tfrecords/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def write_tfrecords(dataset, output_path, num_shards=1):
    """
    Convert training samples to tfrecords

    Args:
      dataset: Dataset as a iterator in (tfds format).
      output_path: Directory to store the tfrecords.
      num_shards: Split the tfrecords to sepecific number of shards.
    """

    writers = [
        tf.io.TFRecordWriter(output_path + "-%05d-of-%05d.tfrecord" % (i, num_shards))
        for i in range(num_shards)
    ]
    for idx, record in enumerate(dataset):
        if idx % 100 == 0:
            logger.info("On image %d", idx)
        seq_example = process_record(record)
        writers[idx % num_shards].write(seq_example.SerializeToString())


def main():
    logger.debug("Successfully initialized video_ssl libraries.")

    URL = "https://storage.googleapis.com/thumos14_files/UCF101_videos.zip"
    download_dir = pathlib.Path(RESOURCES_DATA, "UCF101_subset")
    if not os.path.exists(download_dir.as_posix()):
        subset_paths = download_ufc_101_subset(
            URL,
            num_classes=10,
            splits={"train": 40, "val": 10, "test": 10},
            download_dir=download_dir,
        )
        logger.info("Successfully downloaded UCF101.")
    else:
        subset_paths = {
            mode: (download_dir / mode) for mode in ["train", "val", "test"]
        }
        logger.info(f"Found UCF101 at {download_dir}.")

    logger.debug(subset_paths)
    output_dir = os.path.join(download_dir, "records")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    n_frames = 10
    CLASSES = sorted(os.listdir(os.path.join(download_dir, "train")))

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8, name="image"),
        tf.TensorSpec(shape=(), dtype=tf.int16, name="label"),
    )

    train_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths["train"], n_frames, training=True),
        output_signature=output_signature,
    )
    write_tfrecords(train_ds, os.path.join(output_dir, "train"), num_shards=10)
    val_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths["val"], n_frames), output_signature=output_signature
    )
    write_tfrecords(val_ds, os.path.join(output_dir, "val"), num_shards=5)
    test_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths["test"], n_frames),
        output_signature=output_signature,
    )
    write_tfrecords(test_ds, os.path.join(output_dir, "test"), num_shards=5)

    # Adapt configuration to UCF101 dataset.
    override_for_ucf101(n_frames)
    logger.info("Done with configuration (overwrote from Kinetics-600).")
