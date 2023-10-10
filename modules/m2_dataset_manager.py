
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modules.default import ARG_TRAINING_SET_FPATH, TEST_SIZE, SEED, ARG_TEST_SET_FPATH, PADDING_MAXLEN


class DatasetManager:
    """
    Manages datasets for loading, padding, splitting, and access to train, validation, and test sets.

    Attributes:
        X_train (np.ndarray): Padded training data.
        X_val (np.ndarray): Padded validation data.
        y_train (np.ndarray): Labels for training data.
        y_val_encoded (np.ndarray): One-hot encoded labels for validation data.
        X_test (np.ndarray): Padded test data.
        y_test_encoded (np.ndarray): One-hot encoded labels for test data.

    Example:
        args = {'ARG_TRAINING_SET_FPATH': 'path/to/training.csv', 'ARG_TEST_SET_FPATH': 'path/to/test.csv'}
        data_manager = DatasetManager(args)
        print(data_manager.X_train.shape)
    """

    def __init__(self, args: dict):

        self.train_fpath = args[ARG_TRAINING_SET_FPATH]
        self.test_fpath = args[ARG_TEST_SET_FPATH]

        self._X_train = None
        self._X_val = None
        self._y_train = None
        self._y_val_encoded = None
        self._X_test = None
        self._y_test_encoded = None

    @property
    def X_train(self) -> np.ndarray:
        """Returns the training data after loading and processing if not already loaded."""
        if self._X_train is None:
            self.load_training_data()
        return self._X_train

    @property
    def X_val(self) -> np.ndarray:
        """Returns the validation data after loading and processing if not already loaded."""
        if self._X_val is None:
            self.load_training_data()
        return self._X_val

    @property
    def y_train(self) -> np.ndarray:
        """Returns the training labels after loading and processing if not already loaded."""
        if self._y_train is None:
            self.load_training_data()
        return self._y_train

    @property
    def y_val_encoded(self) -> np.ndarray:
        """Returns the one-hot validation labels after loading and processing if not already loaded."""
        if self._y_val_encoded is None:
            self.load_training_data()
        return self._y_val_encoded

    @property
    def X_test(self) -> np.ndarray:
        """Returns the test data after loading and processing if not already loaded."""
        if self._X_test is None:
            self.load_test_data()
        return self._X_test

    @property
    def y_test_encoded(self) -> np.ndarray:
        """Returns the one-hot test labels after loading and processing if not already loaded."""
        if self._y_test_encoded is None:
            self.load_test_data()
        return self._y_test_encoded

    def __repr__(self):
        return f"<DatasetManager with training data shape {self.X_train.shape}, " \
               f"validation data shape {self.X_val.shape}, and test data shape {self.X_test.shape}>"

    def __str__(self):
        return (f"DatasetManager:\n"
                f"Training data shape: {self.X_train.shape}\n"
                f"Validation data shape: {self.X_val.shape}\n"
                f"Test data shape: {self.X_test.shape}")

    def _load_and_process_data(self, fpath: str):
        """
        Loads and processes a CSV dataset from the specified file path.

        Parameters:
        - fpath (str): Path to the CSV dataset.

        Returns:
        - tuple (np.ndarray, np.ndarray): Padded features and corresponding labels.

        Raises:
        - FileNotFoundError: If the provided file path does not exist.
        - ValueError: If the CSV file has incorrect formatting.
        - Exception: For other errors during the data loading process.
        """

        try:
            data_set = np.loadtxt(fpath, delimiter=',', dtype=np.float32)

            X = data_set[:, :-1]
            y = data_set[:, -1].astype(int)

            # Rationale for padding:
            # 1. Pooling Layers Compatibility: Ensures sequence length doesn't result in non-integral dimensions.
            # 2. Easier Dimension Management: Simplifies tracking sizes across the network.
            X_padded = tf.keras.utils.pad_sequences(X, maxlen=PADDING_MAXLEN, padding='post', dtype='float32')

            return X_padded, y

        except FileNotFoundError:
            raise FileNotFoundError(f"The file at path {fpath} was not found.") from None

        except ValueError:
            raise ValueError(f"The CSV file at path {fpath} seems to be formatted incorrectly.") from None

        except Exception as e:
            raise Exception(f"An error occurred while loading the data from path {fpath}: {str(e)}") from None

    def unload_training_data(self) -> None:
        """
        Unloads (clears) the training data from the instance.

        This method is particularly useful to free up memory, especially when the datasets
        are large and the instance of the DatasetManager is no longer in use.
        """

        self._X_train = None
        self._y_train = None

    def unload_validation_data(self) -> None:
        """
        Unloads (clears) the validation data from the instance.

        This method is particularly useful to free up memory, especially when the datasets
        are large and the instance of the DatasetManager is no longer in use.
        """

        self._X_val = None
        self._y_val_encoded = None

    def load_training_data(self) -> None:
        """
        Loads, processes, and splits the training dataset into training and validation sets.

        This method also encodes the labels into one-hot encoded format for both training
        and validation datasets.

        Note:
            The method uses the training file path provided during the instantiation of
            the DatasetManager and leverages the sklearn's train_test_split to create
            stratified training and validation datasets.
            The split ratio and randomness seed are taken from global constants.
        """

        X_padded, y = self._load_and_process_data(self.train_fpath)

        # Given that the model has to be trained, it's more straightforward for the validation data
        # to represent the original data distribution. So, split the original training data into
        # training and validation sets
        self._X_train, self._X_val, self._y_train, y_val = train_test_split(
            X_padded, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
        )
        self._y_val_encoded = tf.keras.utils.to_categorical(y_val)
        self._y_train_encoded = tf.keras.utils.to_categorical(self._y_train)

    def load_test_data(self) -> None:
        """
        Loads and processes the test dataset.

        This method also encodes the test labels into one-hot encoded format.

        Note:
            The method uses the test file path provided during the instantiation
            of the DatasetManager.
        """

        self._X_test, y_test = self._load_and_process_data(self.test_fpath)
        self._y_test_encoded = tf.keras.utils.to_categorical(y_test)

