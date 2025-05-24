# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt
from tensorflow import keras

# Importing specific modules from libraries
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Importing User-defined functions from utilis
from main_utilis import read_data, data_from_to_numpy, std_normalise_data
from equi_nn_utilis import (
    EquivariantHiddenLayer,
    EquivariantOutputLayer,
    eq_build_model,
)

# Setting numpy print options to display floating point numbers up to 3 decimal points
np.set_printoptions(precision=3, suppress=True)

custom_objects = {
    "EquivariantOutputLayer": EquivariantOutputLayer,
    "EquivariantHiddenLayer": EquivariantHiddenLayer,
}

def standard_model(hp):
    model_sm = Sequential()

    # Define number of layers
    num_layers = hp.Int("num_layers", min_value=1, max_value=4, step=1)

    # Define layer size range
    layer_size_range = hp.Int("layer_size_range", min_value=32, max_value=512, step=16)

    # Add input layer
    model_sm.add(Dense(units=hp.Int("input_units", min_value=layer_size_range, max_value=layer_size_range*2, step=16),
                    input_shape=(4,),
                    activation=hp.Choice("input_activation", values=["relu", "tanh","sigmoid"])))

    # Add hidden layers
    for i in range(num_layers):
        model_sm.add(Dense(units=hp.Int(f"hidden_{i+1}_units", min_value=layer_size_range, max_value=layer_size_range*2, step=16),
                        activation=hp.Choice(f"hidden_{i+1}_activation", values=["relu", "tanh","sigmoid"])))

    # Add output layer
    model_sm.add(Dense(units=2, activation="softmax"))

    # Set optimizer and learning rate
    optimizer = hp.Choice("optimizer", values=["adam", "adagrad", "rmsprop"])
    learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Set loss function
    loss_func = hp.Choice("loss", values=["mean_squared_error"])

    # Set batch size and number of epochs
    batch_size = hp.Int("batch_size", min_value=16, max_value=256, step=16)
    num_epochs = hp.Int("num_epochs", min_value=10, max_value=200, step=10)
    

    model_sm.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=["accuracy", "mean_absolute_error"])

    return model_sm

def augment_data(x_train, y_train):
    # initialize empty lists for augmented data and labels
    x_train_aug = []
    y_train_aug = []
    
    for i in range(len(x_train)):
        ### 0000000000000000000000000
        x_train_aug.append(x_train[i])
        y_train_aug.append(y_train[i])

         ####### 11111111111111111
        # 90 degree clockwise rotation
        x_train_aug.append(
            (
                np.array([x_train[i][3], x_train[i][0], x_train[i][1], x_train[i][2]])
            )
        )
        y_train_aug.append(np.array([y_train[i][1], -y_train[i][0]]))
        
        
       #### 2222222222222222
        # 180 degree clockwise rotation
        x_train_aug.append(
            (
                np.array([x_train[i][2], x_train[i][3], x_train[i][0], x_train[i][1]])
            )
        )
        y_train_aug.append((np.array([-y_train[i][0], -y_train[i][1]])))
        
        ##### 3333333333333333
        # 270 degree clockwise rotation
        x_train_aug.append(
            (
                np.array([x_train[i][1], x_train[i][2], x_train[i][3], x_train[i][0]])
            )
        )
        y_train_aug.append((np.array([-y_train[i][1], y_train[i][0]])))
        
        ### 0000000000000000000000000
        # vertical flip
        x_train_aug.append(
            np.flip(
                np.array([x_train[i][0], x_train[i][1], x_train[i][2], x_train[i][3]])[[2,3,0,1]]
            )
        )
#         y_train_aug.append(np.flip(np.array([y_train[i][0], y_train[i][1]])))
#         y_train_aug.append(np.flip(np.array([y_train[i][1], -y_train[i][0]])))
        y_train_aug.append(np.flip(np.array([-y_train[i][1], y_train[i][0]])))

        ####### 11111111111111111
        # horizontal flip + 90 degree clockwise rotation
        x_train_aug.append(
            np.flip(
              np.array([x_train[i][3], x_train[i][0], x_train[i][1], x_train[i][2]])
            )[[2,3,0,1]]
        )
#         y_train_aug.append(np.flip(np.array([y_train[i][1], -y_train[i][0]])))
#         y_train_aug.append(np.flip(np.array([y_train[i][0], y_train[i][1]])))
#         y_train_aug.append(np.flip(np.array([-y_train[i][0], -y_train[i][1]])))
        y_train_aug.append(np.flip(np.array([y_train[i][0], y_train[i][1]])))
        
        #### 2222222222222222
        # horizontal flip + 180 degree clockwise rotation
        x_train_aug.append(
            np.flip(
                np.array([x_train[i][2], x_train[i][3], x_train[i][0], x_train[i][1]])
            )[[2,3,0,1]]
        )
#         y_train_aug.append(np.flip(np.array([-y_train[i][0], -y_train[i][1]])))
#         y_train_aug.append(np.flip(np.array([-y_train[i][1], y_train[i][0]])))
        y_train_aug.append(np.flip(np.array([y_train[i][1], -y_train[i][0]])))
        
        ##### 3333333333333333
        # horizontal flip + 270 degree clockwise rotation
        x_train_aug.append(
            np.flip(
                np.array([x_train[i][1], x_train[i][2], x_train[i][3], x_train[i][0]])
            )[[2,3,0,1]]
        )
#         y_train_aug.append(np.flip(np.array([-y_train[i][1], y_train[i][0]])))
#         y_train_aug.append(np.flip(np.array([-y_train[i][0], -y_train[i][1]])))
#         y_train_aug.append(np.flip(np.array([y_train[i][0], y_train[i][1]])))
        y_train_aug.append(np.flip(np.array([-y_train[i][0], -y_train[i][1]])))
    return x_train_aug, y_train_aug

# def augment_data(x_train, y_train):
#     # initialize empty lists for augmented data and labels
#     x_train_aug = []
#     y_train_aug = []

#     for i in range(len(x_train)):
#         x_train_aug.append(x_train[i])
#         y_train_aug.append(y_train[i])

#         # 90 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array([-x_train[i][1], x_train[i][0], -x_train[i][3], x_train[i][2]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([y_train[i][1], -y_train[i][0]])))

#         # 180 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array([-x_train[i][2], -x_train[i][3], x_train[i][0], x_train[i][1]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([-y_train[i][0], -y_train[i][1]])))

#         # 270 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array([x_train[i][3], -x_train[i][2], x_train[i][1], -x_train[i][0]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([-y_train[i][1], y_train[i][0]])))

#         # vertical flip
#         x_train_aug.append(
#             np.flip(
#                 np.array([x_train[i][2], x_train[i][3], x_train[i][0], x_train[i][1]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([-y_train[i][0], y_train[i][1]])))

#         # horizontal flip + 90 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array(
#                     [-x_train[i][1], -x_train[i][0], -x_train[i][3], -x_train[i][2]]
#                 )
#             )
#         )
#         y_train_aug.append(np.flip(np.array([-y_train[i][1], y_train[i][0]])))

#         # horizontal flip + 180 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array([-x_train[i][2], -x_train[i][1], x_train[i][0], x_train[i][3]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([y_train[i][0], y_train[i][1]])))

#         # horizontal flip + 270 degree clockwise rotation
#         x_train_aug.append(
#             np.flip(
#                 np.array([x_train[i][3], x_train[i][2], -x_train[i][1], x_train[i][0]])
#             )
#         )
#         y_train_aug.append(np.flip(np.array([y_train[i][1], -y_train[i][0]])))
#     return x_train_aug, y_train_aug

def load_and_evaluate(which_model, features, target):
    """
    This function takes the name of the model and evalutes the models with validation data/ replace validation data with Test Data to evalute test data
    Choose option to evaluate a particular model
    1. model_standard_100.h5
    2. model_standard_1000.h5
    3. model_standard_100_augmented.h5
    4. model_standard_1000_augmented.h5
    5. model_equivariant_100.h5
    6. model_equivariant_1000.h5
    7. model_equivariant_100_augmented.h5
    8. model_equivariant_1000_augmented.h5
    To evaluate call the model and choose the option and pass the dataset that is required to be evaluated
    features = X_val or X_test
    target = y_val or y_test
    """

    if which_model == 1:
        load_model = tf.keras.models.load_model("models/model_standard_100.h5")
        print(
            f"The model is trained for 100 Samples without augmentation using standard model definition"
        )
    elif which_model == 2:
        load_model = tf.keras.models.load_model("models/model_standard_1000.h5")
        print(
            f"The model is trained for 1000 Samples without augmentation using standard model definition"
        )
    elif which_model == 3:
        load_model = tf.keras.models.load_model(
            "models/model_standard_100_augmented.h5"
        )
        print(
            f"The model is trained for 100 Samples with augmentation using standard model definition"
        )
    elif which_model == 4:
        load_model = tf.keras.models.load_model(
            "models/model_standard_1000_augmented.h5"
        )
        print(
            f"The model is trained for 1000 Samples with augmentation using standard model definition"
        )
    elif which_model == 5:
        load_model = tf.keras.models.load_model(
            "models/model_equivariant_100.h5", custom_objects=custom_objects
        )
        print(
            f"The model is trained for 100 Samples without augmentation using Equivariant model definition"
        )
    elif which_model == 6:
        load_model = tf.keras.models.load_model(
            "models/model_equivariant_1000.h5", custom_objects=custom_objects
        )
        print(
            f"The model is trained for 1000 Samples without augmentation using Equivariant model definition"
        )
    elif which_model == 7:
        load_model = tf.keras.models.load_model(
            "models/model_equivariant_100_augmented.h5", custom_objects=custom_objects
        )
        print(
            f"The model is trained for 100 Samples with augmentation using Equivariant model definition"
        )
    else:
        load_model = tf.keras.models.load_model(
            "models/model_equivariant_1000_augmented.h5", custom_objects=custom_objects
        )
        print(
            f"The model is trained for 1000 Samples with augmentation using Equivariant model definition"
        )

    evaluated_results = load_model.evaluate(features, target)

    return evaluated_results


