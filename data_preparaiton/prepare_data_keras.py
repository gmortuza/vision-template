import os
from read_parameter import Params
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def get_data_keras(params):
    dataset_dir = params.dataset_dir
    train_dir = os.path.join(dataset_dir, 'train')
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # This method is very slow. Most of the training time is wasted on the input preparation
    # Better to use tf.Data
    data_generator = ImageDataGenerator(
        rescale=1./255,  # Normalize the image data
        shear_range=0.2,
        zoom_range=.2,
        validation_split=params.validation_rate,
        horizontal_flip=True
    )
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if params.label_from == 'directory':
        train_generator = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=params.image_size,
            batch_size=params.hparam["batch_size"],
            subset="training",
            validation_split=params.validation_rate,
            seed=1234  # reproducibility
        ).cache().prefetch(buffer_size=AUTOTUNE)
        validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=params.image_size,
            batch_size=params.hparam["batch_size"],
            subset="validation",
            validation_split=params.validation_rate,
            seed=1234
        ).cache().prefetch(buffer_size=AUTOTUNE)

        return train_generator, validation_generator

    elif params.label_from == 'dataframe':
        pass


if __name__ == '__main__':
    param = Params("../config.json")
    get_data_keras(param)
