import json
from read_parameter import Params
from models.model import get_model
from data_preparaiton.prepare_data_keras import get_data_keras
from data_preparaiton.prepare_data_custom import get_data_custom
from models.callbacks import get_callbacks
import tensorflow as tf
import os

tf.autograph.set_verbosity(0)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


def log_device_information(logger):
    # get the gpu information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) >= 1:
        logger.info(f"Total {len(gpus)} GPUs found")
    else:
        logger.warning("No GPU found. It will train on CPU")


def train_model(params):
    logger = params.logger
    log_device_information(logger)
    logger.info("building the model")
    model = get_model(params)
    logger.info("model building done")
    logger.info("Getting the input")
    if params.data_preparation == "keras":
        train_generator, validation_generator = get_data_keras(params)
    else:
        # Custom data preparation
        train_generator, validation_generator = get_data_custom(params)
    logger.info("Inputs are ready")
    logger.info("Training the model")
    history = model.fit(
        train_generator,
        steps_per_epoch=None,
        epochs=int(params.hparam["epochs"]),
        verbose=params.verbose,
        validation_data=validation_generator,
        callbacks=get_callbacks(params)
    )
    logger.info("Saving the final model and it's result")
    # If it is a hyper parameter search we won't save the model
    if not params.is_hyper_parameter_searching:
        # If there exists a model already we will remove that
        final_model_path = os.path.join(params.output_dir, "final_model")
        if os.path.isdir(final_model_path):
            os.rmdir(final_model_path)
        os.makedirs(final_model_path)
        # This will save entire model
        model.save(final_model_path)
    with open(os.path.join(params.output_dir, "train_history.json"), "w") as train_history_file:
        train_history_file.write(json.dumps(history.history))
    logger.info("Finish writing the history and model")
    return history


if __name__ == '__main__':
    parameters = Params("config.yaml")
    train_model(params=parameters)
