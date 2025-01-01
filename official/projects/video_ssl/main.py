# -- LIBRARY IMPORTS & PREPARTION --

from utils import logger, RESOURCES_DATA, RESOURCES_EVALS
from .prepare import *
from .config import override_for_ucf101

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

    # Adapt configuration to UCF101 dataset and pretty-print.
    WIDTH, HEIGHT = 224, 224
    exp_config = override_for_ucf101(n_frames, WIDTH, HEIGHT)
    # Pretty-print current config.
    pp = pprint.PrettyPrinter(indent=4)
    # logger.debug(pp.pformat(exp_config))
    logger.info("Done with configuration (overwrote from Kinetics-600).")

    # Try detecting the hardware.
    try:
        tpu_resolver = (
            tf.distribute.cluster_resolver.TPUClusterResolver()
        )  # TPU detection
    except ValueError:
        tpu_resolver = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy.
    if tpu_resolver:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
        logger.debug("Running on TPU ", tpu_resolver.cluster_spec().as_dict()["worker"])
    elif len(gpus) > 1:
        distribution_strategy = tf.distribute.MirroredStrategy(
            [gpu.name for gpu in gpus]
        )
        logger.debug("Running on multiple GPUs ", [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        distribution_strategy = tf.distribute.get_strategy()
        # Default strategy that works on CPU and single GPU.
        logger.debug("Running on single GPU ", gpus[0].name)
    else:
        # Default strategy that works on CPU and single GPU.
        distribution_strategy = tf.distribute.get_strategy()
        logger.debug("Running on CPU")
    logger.debug(
        "Number of accelerators: " + str(distribution_strategy.num_replicas_in_sync)
    )

    # Performing the adctual training.
    model_dir = pathlib.Path(RESOURCES_EVALS, "trained_model")
    export_dir = pathlib.Path(RESOURCES_EVALS, "exported_model")
    with distribution_strategy.scope():
        task = task_factory.get_task(exp_config.task, logging_dir=model_dir)

    # Traing and export model.
    model, eval_logs = train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode="train_and_eval",
        params=exp_config,
        model_dir=model_dir,
    )
    export_saved_model_lib.export_inference_graph(
        input_type="image_tensor",
        batch_size=1,
        input_image_size=[n_frames, HEIGHT, WIDTH],
        params=exp_config,
        checkpoint_path=tf.train.latest_checkpoint(model_dir),
        export_dir=export_dir,
    )

    # Import exported model and run predictions.
    imported = tf.saved_model.load(export_dir)
    model_fn = imported.signatures["serving_default"]

    frames, label = list(test_ds.shuffle(buffer_size=90).take(1))[0]
    frames = tf.expand_dims(frames, axis=0)
    result = model_fn(frames)
    predicted_label = tf.argmax(result["probs"][0])
    logger.info(f"Actual: {CLASSES[label]}")
    logger.info(f"Predicted: {CLASSES[predicted_label]}")
    imageio.mimsave(
        pathlib.Path(RESOURCES_EVALS, "animation.gif").as_posix(), frames[0], fps=10
    )
