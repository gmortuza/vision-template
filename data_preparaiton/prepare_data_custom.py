# keras.ImageDataGenerator doesn't provide much flexibility
import tensorflow as tf
import numpy as np
import os
import pathlib
from read_parameter import Params

AUTOTUNE = tf.data.experimental.AUTOTUNE

# TODO: data from dataframe
# TODO: introduce more data augmentation


class PrepareData:

    def __init__(self, parameters):
        self.params = parameters
        train_dir = pathlib.Path(self.params.dataset_dir).joinpath("train")
        self.class_names = self._get_class_name(train_dir)
        self.params.logger.info(f"Total {str(self.class_names.shape[0])} classes found in the "
                                f"{str(train_dir)} directory")
        if parameters.label_from == 'directory':
            self.train_ds, self.val_ds = self._prepare_data_from_directory(train_dir)
        else:
            self.params.logger.error("Right now label is extracted only from directory")

    def flow_from_directory(self, subset="training"):
        if subset == 'validation':
            return self.val_ds
        else:
            return self.train_ds

    def flow_from_df(self, subset="training"):
        pass

    def _prepare_data_from_directory(self):
        pass

    def _prepare_data_from_directory(self, train_dir):
        logger = self.params.logger
        # list of the dataset
        # @@@ Depending on the data directory format we might need to modify this
        list_ds = tf.data.Dataset.list_files(str(train_dir / '*/*'), shuffle=False)
        # Total number of training data
        total_training_example = tf.data.experimental.cardinality(list_ds).numpy()
        logger.info(f"Total {str(total_training_example)} found the training folder")
        # Total number of data that will be used for validation
        val_size = int(total_training_example * self.params.validation_rate)
        train_size = total_training_example - val_size
        logger.info(f"Total {train_size} images will be used for training")
        logger.info(f"Total {val_size} images will be used for validation")
        # before splitting validation and train it should be shuffled
        list_ds = list_ds.shuffle(total_training_example, reshuffle_each_iteration=False)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)
        train_ds = train_ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        train_ds = self._configure_for_performance(train_ds)
        val_ds = val_ds.map(self._process_path, num_parallel_calls=AUTOTUNE)
        val_ds = self._configure_for_performance(val_ds)
        return train_ds, val_ds

    def _get_class_name(self, train_dir):
        # @@@ Some times class can be in different format. Depending on that we might need to modify this method
        return np.array(sorted([item.name for item in train_dir.glob("*") if os.path.isdir(item)]))

    def _get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = parts[-2] == self.class_names
        # Integer encode the label
        return tf.argmax(one_hot)

    def _decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=self.params.num_channel)
        # Normalize image
        img = tf.image.per_image_standardization(img)
        # resize the image to the desired size
        return tf.image.resize(img, self.params.image_size)

    def _process_path(self, file_path):
        label = self._get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self._decode_img(img)
        return img, label

    def _configure_for_performance(self, ds):
        # If the dataset is too big then it
        if self.params.data_set_cache == "memory":
            ds = ds.cache()
        else:
            ds = ds.cache("./tfcache")
        # There will always be this amount of data in the buffer
        # For perfect shuffling, set the buffer size equal to the full size of the dataset.
        ds = ds.shuffle(buffer_size=1000, seed=1234)
        ds = ds.repeat(3)
        ds = ds.batch(self.params.hparam["batch_size"])
        # This will make sure few of the data(preferably at least one batch) is ready before
        # AUTOTUNE will dynamically calcualate about how much element will be prefetched
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    # train_ds = configure_for_performance(train_ds)
    # val_ds = configure_for_performance(val_ds)


def get_data_custom(params):
    data = PrepareData(params)
    return data.train_ds, data.val_ds


if __name__ == '__main__':
    params = Params("../config.json")
    ds = PrepareData(params)
