import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import ParameterGrid
from read_parameter import Params
from train import train_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_accuracy(parameters):
    metrics = train_model(parameters)
    return metrics.history["accuracy"][-1], metrics.history["val_accuracy"][-1]


def search_hyper_parameter(parameters, hparams):
    logging_file_name = ""
    for param in hparams:
        logging_file_name += "_" + param + "_" + str(hparams[param])
    logging_file_loc = os.path.join(parameters.log_dir, 'hparam', logging_file_name)

    with tf.summary.create_file_writer(logging_file_loc).as_default():
        hp.hparams(hparams)
        accuracy, val_accuracy = get_accuracy(parameters)
        tf.summary.scalar("accuracy", accuracy, step=1)
        tf.summary.scalar("val_accuracy", val_accuracy, step=1)


def main():
    parameters = Params("config.yaml")
    parameters.is_hyper_parameter_searching = True
    param_grids = list(ParameterGrid(parameters.hparam_tuning))
    for hparams in param_grids:
        # update the parameters with these hyper parameters
        parameters.hparam = hparams
        # Change
        search_hyper_parameter(parameters, hparams)


if __name__ == '__main__':
    main()
