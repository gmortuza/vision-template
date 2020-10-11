from tensorflow.keras import callbacks
import os


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
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            # profile_batch=[50, 70]
        ))

    return callback_collections
