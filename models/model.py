from models.base_model import get_base_model
import os
from tensorflow.train import latest_checkpoint


def get_model(params):
    params.logger.info("Getting the base model")
    model = get_base_model(params)
    # if checkpoint already have a saved model we will use that
    if os.path.isfile(os.path.join(params.checkpoint_dir, "checkpoint")) and not params.is_hyper_parameter_searching:
        params.logger.info("Checkpoint model found. Loading the weights from there.")
        model.load_weights(latest_checkpoint(params.checkpoint_dir))
    return model

