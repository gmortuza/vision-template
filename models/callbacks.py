from tensorflow.keras import callbacks
import os


class CustomCallback(callbacks.Callback):
    """
    Available method:
    on_{train|test|predict}_{begin|end}(self, logs=None)
    on_{train|test|predict}_batch_{begin|end}(self, batch, logs=None)
    on_epoch_{begin|end}(self, epoch, logs=None)
    example: on_train_begin(self, logs=None)
    access the model using self.model
    Set self.model.stop_training = True to immediately interrupt training.
    self.model.optimizer.learning_rate --> change learning rate

    more on: https://www.tensorflow.org/guide/keras/custom_callback
    """
    def __init__(self):
        super(CustomCallback, self).__init__()
    pass

# TODO: add more callbacks


def get_callbacks(params):
    callback_collections = []
    if "checkpoint" in params.callbacks:
        # if checkpoint directory not exists then we will create a checkpoint directory
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        callback_collections.append(callbacks.ModelCheckpoint(
            filepath=os.path.join(params.checkpoint_dir, "checkpoint"),
            monitor="val_accuracy",
            verbose=int(params.verbose),
            save_best_only=True,
            save_weights_only=True,
        ))
    if "reduce_lr_plateau" in params.callbacks:
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
        callback_collections.append(callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=.2,
            patience=5,
            # min_lr lower bound of learning rate
        ))
    if "tensorboard" in params.callbacks:
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
        callback_collections.append(callbacks.TensorBoard(
            log_dir=params.log_dir,
            histogram_freq=1,
            embeddings_freq=1,
            write_graph=True,
            write_images=True,
            profile_batch=[50, 70]  # profile from batch 50 to 70
        ))
    if "custom" in params.callbacks:
        callback_collections.append(CustomCallback())

    return callback_collections
