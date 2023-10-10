
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modules.default import (
    OVERSAMPLING_TECHNIQUES,
    NUM_EPOCHS, BATCH_SIZE,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_REDUCE_LR,
    INITIAL_WEIGHTS, TRAINING_SUFFIX,
    SEED
)


class ModelTrainer:
    """
    A class to train TensorFlow/Keras models, apply callbacks, and plot performance metrics.
F
    Attributes:
    - models_dict: Dictionary containing the models to be trained.
    - models_output: Directory for models to be saved.
    """

    def __init__(self, args: dict,
                 X_train_resampled_dict: dict,
                 y_train_resampled_encoded_dict: dict,
                 X_val: np.ndarray,
                 y_val_encoded: np.ndarray,
                 model_architectures_dict: dict):
        """
        Initialises the ModelTrainer.

        Args:
            model_architectures_dict: A dictionary containing the models to be trained.
            args: A dictionary containing the directory for models to be saved.
        """
        self.args = args
        self.X_train_resampled_dict = X_train_resampled_dict
        self.y_train_resampled_encoded_dict = y_train_resampled_encoded_dict
        self.X_val = X_val
        self.y_val_encoded = y_val_encoded
        self.model_architectures_dict = model_architectures_dict

    def _get_callbacks(self, model_name, oversampling_technique) -> list:
        """Prepare a list of callbacks for training."""

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=PATIENCE_EARLY_STOPPING,
                                                          verbose=1,
                                                          restore_best_weights=True)
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=PATIENCE_REDUCE_LR,
                                                         verbose=1)

        callbacks = [early_stopping, reduce_lr]

        best_model_path = os.path.join(self.args[model_name][f'{oversampling_technique}{TRAINING_SUFFIX}'],
                                       f'best_{model_name}_{oversampling_technique}.h5')

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss',
                                                              save_best_only=True, mode='min', verbose=1)
        callbacks.append(model_checkpoint)

        csv_logger_path = os.path.join(self.args[model_name][f'{oversampling_technique}{TRAINING_SUFFIX}'],
                                       f'training_log_{model_name}_{oversampling_technique}.csv')

        csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path, separator=',', append=False)

        callbacks.append(csv_logger)

        return callbacks

    def _plot_perform_metrics(self, history, oversampling_technique, model_name) -> None:
        """
        Plots the performance metrics for the trained models.
        """
        # Extract metrics from the history object
        metrics_to_plot = [metric for metric in history.history.keys()
                           if not metric.startswith('val_') | metric.startswith('lr')]

        for metric in metrics_to_plot:
            plt.figure()
            plt.plot(history.history[metric], label=f"Training {metric}")
            plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
            plt.title(f'{oversampling_technique} | {model_name} | {metric}')
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend()

            path_to_save = os.path.join(self.args[model_name][f'{oversampling_technique}{TRAINING_SUFFIX}'],
                                        f'plot_{model_name}_{oversampling_technique}_{metric}.png')
            plt.savefig(path_to_save)
            plt.close()

    def training_process(self) -> None:
        """
        Train models with provided data.
        """

        if not isinstance(self.X_train_resampled_dict, dict):
            raise TypeError("X_train_resampled_dict must be a dictionary.")

        if not isinstance(self.y_train_resampled_encoded_dict, dict):
            raise TypeError("y_train_resampled_encoded_dict must be a dictionary.")

        for model_name, model_architecture in self.model_architectures_dict.items():

            if model_name not in self.args:
                raise KeyError(f"'{model_name}' not found in the provided args dictionary.")

            print(f"\n - Training {model_name}...")

            print(f"    -> model_architecture memory address: {id(model_architecture)}")

            # To make the various training runs more comparable, keep the initial model's
            # weights in a checkpoint file, and load them into each model before training:
            initial_weights_path = os.path.join(self.args[model_name][f'{INITIAL_WEIGHTS}{TRAINING_SUFFIX}'],
                                                f'initial_weights_{model_name}.h5')

            model_architecture.save_weights(initial_weights_path)

            first_iteration = True

            for oversampling_technique in OVERSAMPLING_TECHNIQUES:

                if oversampling_technique not in self.X_train_resampled_dict:
                    raise KeyError(f"'{oversampling_technique}' not found in the provided X_train_resampled_dict.")

                if oversampling_technique not in self.y_train_resampled_encoded_dict:
                    raise KeyError(f"'{oversampling_technique}' not found in the provided "
                                   f"y_train_resampled_encoded_dict.")

                print(f'\n[ {oversampling_technique} oversampling technique ]')

                # Reproducibility
                np.random.seed(SEED)
                tf.random.set_seed(SEED)

                # Clear any leftovers from previous models, layers, or
                # even callbacks from the TensorFlow backend
                tf.keras.backend.clear_session()

                if not first_iteration:
                    print('getting initial weights...')
                    model_architecture.load_weights(initial_weights_path)
                else:
                    first_iteration = False

                # For each model, check the initial saved weights (parameters) that
                # should be used across all oversampling techniques.
                weights = model_architecture.get_weights()
                print(f'W1: {np.mean(weights[0]): .7f}, b1: {np.mean(weights[1]): .7f}  -  '
                      f'W2: {np.mean(weights[2]): .7f}, b2: {weights[3]}')

                X_train = self.X_train_resampled_dict[oversampling_technique]
                y_train_encoded = self.y_train_resampled_encoded_dict[oversampling_technique]
                print(f'X_mean: {np.mean(X_train): .7f} - y_mean: {np.mean(y_train_encoded): .7f}')

                callbacks = self._get_callbacks(model_name, oversampling_technique)

                history = model_architecture.fit(X_train, y_train_encoded,
                                                 epochs=NUM_EPOCHS,
                                                 batch_size=BATCH_SIZE,
                                                 validation_data=(self.X_val, self.y_val_encoded),
                                                 callbacks=callbacks)

                self._plot_perform_metrics(history, oversampling_technique, model_name)
