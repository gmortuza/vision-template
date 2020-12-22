# This is very basic model so far
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Input, MaxPool2D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow_hub as hub


def get_optimizer(optimizer, lr):
    lr_schedule = optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=100000,  # learning rate will decay after each this amount of steps
        decay_rate=0.96,  # rate at which lr will decay
        staircase=True  # step / decay_steps is an integer division.
    )
    if optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=lr_schedule)
    elif optimizer == 'adagrad':
        opt = optimizers.Adagrad(learning_rate=lr_schedule)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=lr_schedule)
    else:  # Adam is the default one
        opt = optimizers.Adam(learning_rate=lr_schedule)

    return opt


def get_base_model_no_transfer_learning(params):
    activation = params.hparam["activation"]
    dropout = params.hparam["dropout"]
    output_class = params.num_output_class
    inputs = Input(shape=params.input_shape)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=activation,
                   input_shape=params.input_shape)(inputs)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=activation)(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Flatten()(model)
    model = Dense(128, activation=activation)(model)
    model = Dropout(dropout)(model)
    model = Dense(256, activation=activation)(model)
    model = Dropout(dropout)(model)
    model = Dense(256, activation=activation)(model)
    model = Dense(output_class, activation="softmax")(model)

    model = Model(inputs=inputs, outputs=[model])

    return model


def get_pretrained_model(params):
    inputs = Input(shape=params.input_shape)
    model = hub.KerasLayer(params.transfer_learning["hub_link"],
                           trainable=params.transfer_learning["do_fine_tuning"])(inputs)
    model = Dropout(params.hparam["dropout"])(model)
    model = Dense(params.num_output_class, activation="softmax")(model)
    model = Model(inputs=inputs, outputs=[model])

    return model


def get_base_model(params):
    if hasattr(params, "transfer_learning"):
        # return a pretrained model
        model = get_pretrained_model(params)
    else:
        # return a basic model
        model = get_base_model_no_transfer_learning(params)
    # compile the model
    model.compile(
        optimizer=get_optimizer(params.hparam["optimizer"], lr=params.hparam["learning_rate"]),
        loss=params.hparam["loss"],
        metrics=["accuracy"]
    )
    return model
