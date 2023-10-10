
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from typing import Tuple
from modules.default import KEY_SIMPLE_RESAMPLING, KEY_SMOTE_RESAMPLING, KEY_ADASYN_RESAMPLING, SEED


class MultiClassOversampler:
    """
    Provides functionality to oversample multi-class datasets using various techniques.

    Attributes:
    - X_train_resampled_dict: A dictionary containing the resampled training data.
    - y_train_resampled_encoded_dict: A dictionary containing the resampled training labels.

    Methods:
    - _simple_oversample: Performs simple oversampling.
    - _oversample: Orchestrates the various oversampling methods.

    Raises:
    - RuntimeError: If there's an error during the oversampling process.
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialise the MultiClassOversampler.

        Parameters:
        - X_train (np.ndarray): The training data.
        - y_train (np.ndarray): The training labels.
        """

        self._X_train = X_train
        self._y_train = y_train

        self._X_train_resampled_dict = {}
        self._y_train_resampled_encoded_dict = {}

    @property
    def X_train_resampled_dict(self) -> dict:
        """Returns the dictionary containing the resampled training data."""
        return self._X_train_resampled_dict

    @property
    def y_train_resampled_encoded_dict(self) -> dict:
        """Returns the dictionary containing the resampled training labels."""
        return self._y_train_resampled_encoded_dict

    def __repr__(self) -> str:
        shape = self._X_train_resampled_dict.get(KEY_SMOTE_RESAMPLING, None).shape if KEY_SMOTE_RESAMPLING in self._X_train_resampled_dict else 'Unavailable'
        return f"<MultiClassOversampler smote_shape={shape}>"

    def __str__(self) -> str:
        techniques = ", ".join(self._X_train_resampled_dict.keys())
        return f"MultiClassOversampler with techniques: {techniques}"

    def _simple_oversample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs simple oversampling.

        For each class in the training data, it resamples the class to match the number
        of samples in the majority class.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The resampled training data and labels.
        """

        distribution_classes = np.bincount(self._y_train)
        max_samples_majority_class = np.max(distribution_classes)

        X_resampled = []
        y_resampled = []

        for label in np.unique(self._y_train):
            mask = self._y_train == label
            X_train_label = self._X_train[mask]

            if len(X_train_label) < max_samples_majority_class:
                X_train_label_resampled = resample(X_train_label, replace=True, n_samples=max_samples_majority_class, random_state=SEED)
                X_resampled.append(X_train_label_resampled)
                y_resampled.extend([label] * max_samples_majority_class)
            else:
                X_resampled.append(X_train_label)
                y_resampled.extend([label] * len(X_train_label))

        return np.vstack(X_resampled), np.array(y_resampled)

    def oversample(self) -> None:
        """
        Orchestrates the oversampling methods.

        Applies simple oversampling, SMOTE, and ADASYN methods on the training data.
        The resampled data and labels are stored in instance attributes.

        Raises:
        - RuntimeError: If there's an error during the oversampling process.
        """

        try:
            # Simple oversampling
            X_res_simple, y_res_simple = self._simple_oversample()
            self._X_train_resampled_dict[KEY_SIMPLE_RESAMPLING] = X_res_simple
            self._y_train_resampled_encoded_dict[KEY_SIMPLE_RESAMPLING] = tf.keras.utils.to_categorical(y_res_simple)

            # SMOTE oversampling
            sm = SMOTE(random_state=SEED)
            X_res_smote, y_res_smote = sm.fit_resample(self._X_train, self._y_train)
            self._X_train_resampled_dict[KEY_SMOTE_RESAMPLING] = X_res_smote
            self._y_train_resampled_encoded_dict[KEY_SMOTE_RESAMPLING] = tf.keras.utils.to_categorical(y_res_smote)

            # ADASYN oversampling
            ada = ADASYN(random_state=SEED)
            X_res_ada, y_res_ada = ada.fit_resample(self._X_train, self._y_train)
            self._X_train_resampled_dict[KEY_ADASYN_RESAMPLING] = X_res_ada
            self._y_train_resampled_encoded_dict[KEY_ADASYN_RESAMPLING] = tf.keras.utils.to_categorical(y_res_ada)

        except Exception as e:
            raise RuntimeError(f"Error during oversampling: {str(e)}") from e

    def unload_resampled_data(self) -> None:
        """
        Unloads (clears) the resampled training data and labels from the instance.

        This method is particularly useful to free up memory, especially when the resampled datasets
        are large and the instance of the MultiClassOversampler is no longer in use or needs to be reset.
        """
        self._X_train_resampled_dict = {}
        self._y_train_resampled_encoded_dict = {}
