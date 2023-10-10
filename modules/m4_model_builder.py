
import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from typing import Callable
from modules.default import (
    KEY_BASELINE_NN,
    KEY_DEEP_CNN,
    UNITS_DENSE,
    DROPOUT_RATE,
    INITIAL_NUM_FILTERS,
    KERNEL_SIZE,
    NUM_RESIDUAL_BLOCKS,
    COMPILE_MODELS_BOOL
)


class ModelArchitectureBuilder:
    """
    A utility class to construct and manage various model architectures, primarily for classification tasks.

    The class offers functionality to build a baseline neural network and a deep convolutional
    neural network (CNN) with residual blocks. The CNN model is based on:
    - Rajpurkar et al. (2017). Cardiologist-level arrhythmia detection with convolutional
      neural networks. arXiv preprint arXiv:1707.01836.
    - https://github.com/VinGPan/paper_implementations/blob/master/ecg_classification/ecg_classification_model.py
    - https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

    Apart from building these architectures, it also provides utility functions to compile the constructed
    models and maintain a dictionary of them. Moreover, F1 scores for each class can be computed, aiding in
    model evaluation for multi-class classification problems. Additionally, there is functionality to visualise
    the constructed models.

    Attributes:
    - input_size (int): The size of the input data.
    - num_classes (int): The number of target classes in the classification task.
    - models_dict (dict): A dictionary maintaining built models.

    Methods:
    - _f1_score_per_class: Generates a function to calculate the F1 score for a specific class.
    - _get_metrics: Generates a list of metrics, including accuracy and F1 score for each class.
    - _compile_model: Compiles a provided model with specified parameters.
    - _add_model: Adds a new model-building function to the dictionary of models.
    - build_all: Constructs all available model architectures.
    - _build_baseline_model: Constructs a simple baseline neural network model.
    - _pad_depth: Pads the channel dimension of the input tensor to match the desired number.
    - _initial_block: Represents the initial block of the CNN comprising a convolutional layer, batch normalisation, and activation.
    - _residual_block_type_1: Represents the first type of residual block with two convolutional layers.
    - _residual_block_type_2: Represents the second type of residual block used in the deep CNN.
    - _final_block: Represents the final block of the CNN used for classification.
    - _build_deep_cnn_model: Constructs a deep CNN model with the specified input shape and number of classes.
    - visualise_models: Generates plots for each of the built models showcasing their architecture.

    Raises:
    - ValueError: If the provided input size or number of classes is invalid.
                  If the input size for the deep CNN model is not appropriate.
    """

    def __init__(self, input_size: int, num_classes: int):

        self.input_size = input_size
        self.num_classes = num_classes

        # Ensure that input_size and num_classes are valid inputs
        if not isinstance(self.input_size, int):
            raise ValueError('input_size must be of type int.')

        if not isinstance(self.num_classes, int) or self.num_classes <= 0:
            raise ValueError('num_classes must be a positive integer.')

        self.models_dict = {}

    @staticmethod
    def _f1_score_per_class(class_index):
        """
        Generate a function that calculates the F1 score for a specific class.
        This method uses nested functions to dynamically create a function for a specific class index.
        By using nested functions:
        1. The resulting function only requires y_true and y_pred as arguments, aligning with
            Keras' metrics expectations.
        2. The specific class index is "baked in" to the resulting function.
        3. Each generated function is uniquely named based on the class index.

        Args:
        - class_index (int): The index of the class for which the F1 score is to be computed.

        Returns:
        - function: A function that calculates the F1 score for the specified class when given true and predicted labels.
        """

        def f1(y_true, y_pred):
            """
            Calculate the F1 score for the specified class_index.

            Args:
            - y_true (tensor): A tensor of true labels (often one-hot encoded).
            - y_pred (tensor): A tensor of predicted labels (often in terms of probabilities, not hard labels).

            Returns:
            - float: The F1 score for the class specified by class_index.
            """

            # Convert one-hot encoded truth values to label indices
            y_true_labels = K.argmax(y_true, axis=-1)
            y_pred_labels = K.argmax(y_pred, axis=-1)

            # Create masks for the given class
            y_true_mask = K.cast(K.equal(y_true_labels, class_index), 'float32')
            y_pred_mask = K.cast(K.equal(y_pred_labels, class_index), 'float32')

            tp = K.sum(y_true_mask * y_pred_mask)  # true positive
            fp = K.sum((1 - y_true_mask) * y_pred_mask)  # false positive
            fn = K.sum(y_true_mask * (1 - y_pred_mask))  # false negative

            f1_value = (2 * tp) / (2 * tp + fp + fn + K.epsilon())

            return f1_value

        # Name the function for clarity and uniqueness in metrics
        f1.__name__ = f'f1_class_{class_index}'
        return f1

    def _get_metrics(self, num_classes):
        """
        Generate a list of metrics, including accuracy and F1 score for each class.

        Args:
        - num_classes (int): The total number of classes for which individual F1 scores are to be computed.

        Returns:
        - list: A list of metrics where the first metric is 'accuracy' followed by the F1 score functions for each class.
        """

        metrics = ['accuracy']
        for i in range(num_classes):
            metrics.append(self._f1_score_per_class(i))

        return metrics

    def _compile_model(self, model: tf.keras.models.Model):
        """
        Compiles the provided model with specified parameters.

        Args:
            model (tf.keras.models.Model): The uncompiled model.
        """

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=self._get_metrics(self.num_classes))

    def _add_model(self, model_name: str, build_function: Callable) -> None:
        """
        Add a new model building function to the dictionary of models.
        """
        self.models_dict[model_name] = build_function()

    def build_all(self, compile_models: bool = COMPILE_MODELS_BOOL) -> None:
        """
        Build all model architectures.
        """
        self._add_model(model_name=KEY_BASELINE_NN, build_function=self._build_baseline_model)
        self._add_model(model_name=KEY_DEEP_CNN, build_function=self._build_deep_cnn_model)

        if compile_models:
            for model_key, model in self.models_dict.items():
                self._compile_model(model)

    def _build_baseline_model(self) -> tf.keras.Model:
        """
        Constructs a baseline neural network.

        Returns:
            tf.keras.models.Model: The constructed baseline neural network model.
        """

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.input_size,)))
        model.add(tf.keras.layers.Dense(UNITS_DENSE, activation='relu'))
        model.add(tf.keras.layers.Dropout(DROPOUT_RATE))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        return model

    @staticmethod
    def _pad_depth(x: tf.Tensor, desired_channels: int) -> tf.Tensor:
        """
        Pad the channel dimension of the input tensor to match the desired number of channels.

        This function uses a (1x1) convolution to increase the number of channels of the input tensor.

        Args:
            x: Input tensor.
            desired_channels: Desired number of channels.

        Return:
            Tensor with the desired number of channels.
        """

        # The paper does not specify how to handle the dimension mismatch
        # while creating the residual block. Here, we are using (1X1) Convolution
        # filters to match-up the channels

        x = tf.keras.layers.Conv1D(desired_channels, 1, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return x

    @staticmethod
    def _initial_block(input: tf.Tensor, k: int) -> tf.Tensor:
        """
        Initial block comprising a convolutional layer, batch normalisation, and activation.

        Args:
            input: Input tensor.
            k: Scaling factor for the number of filters.

        Return:
            Processed tensor.
        """
        x = tf.keras.layers.Conv1D(filters=INITIAL_NUM_FILTERS * k, kernel_size=KERNEL_SIZE, padding='same',
                                   kernel_initializer='he_normal')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        return x

    @staticmethod
    def _residual_block_type_1(input: tf.Tensor, k: int) -> tf.Tensor:
        """
        First type of residual block with two convolutional layers.

        Args:
            input: Input tensor.
            k: Scaling factor for the number of filters.

        Return:
            Residual connection output tensor.
        """

        x = tf.keras.layers.Conv1D(filters=INITIAL_NUM_FILTERS * k,
                                   kernel_size=KERNEL_SIZE, padding='same',
                                   kernel_initializer='he_normal')(input)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Conv1D(filters=INITIAL_NUM_FILTERS * k,
                                   kernel_size=KERNEL_SIZE, padding='same',
                                   kernel_initializer='he_normal')(x)

        x = tf.keras.layers.Add()([x, input])

        return x

    def _residual_block_type_2(self, input: tf.Tensor, k: int, subsample: bool, pad_channels: bool) -> tf.Tensor:
        """
        Second type of residual block used for blocks from 2 to 16.

        Args:
            input: Input tensor.
            k: Scaling factor for the number of filters.
            subsample: Boolean, whether to subsample the input or not.
            pad_channels: Boolean, whether to pad the channels or not.

        Return:
            Residual connection output tensor.
        """

        short_cut = input

        # Subsample Input using max pooling. Subsampling is done every alternate block
        if subsample:
            input = tf.keras.layers.MaxPool1D(2, 2)(input)
            # When a residual block subsamples the input, the corresponding shortcut connections also subsample
            # their input using a Max Pooling operation with the same subsample factor
            short_cut = input

        # Whenever k increases we need to pad the 'shortcut', else channel dimensions do not match
        if pad_channels:
            short_cut = self._pad_depth(input, INITIAL_NUM_FILTERS * k)

        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Conv1D(filters=INITIAL_NUM_FILTERS * k,
                                   kernel_size=KERNEL_SIZE, padding='same',
                                   kernel_initializer='he_normal')(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Conv1D(filters=INITIAL_NUM_FILTERS * k,
                                   kernel_size=KERNEL_SIZE, padding='same',
                                   kernel_initializer='he_normal')(x)

        x = tf.keras.layers.Add()([x, short_cut])

        return x

    @staticmethod
    def _final_block(input: tf.Tensor, num_classes: int) -> tf.Tensor:
        """
        Final block for classification comprising batch normalisation,
        activation, flattening, dense layer, and softmax.

        Args:
            input: Input tensor.
            num_classes: Number of output classes.

        Return:
            Processed tensor with probabilities for each class.
        """

        x = tf.keras.layers.BatchNormalization()(input)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(num_classes)(x)
        x = tf.keras.layers.Activation('softmax')(x)

        return x

    def _build_deep_cnn_model(self) -> tf.keras.Model:
        """
        Constructs a deep CNN model with the specified input shape and number of classes.

        This model consists of an initial block followed by residual blocks and a final block.

        Returns:
        - tf.keras.Model: Constructed deep CNN model.
        """

        # When max-pooling is involved, it's crucial to design the architecture in a way
        # that avoids fractional divisions (non-integer). The choice of kernel size, stride,
        # and padding can also influence the effective size of the output feature map. If
        # these are not chosen wisely, the resulting dimensions after convolution might not
        # be whole numbers.
        if not self.input_size == 200:
            raise ValueError('Input size should be at least 200 because there are max-pooling layers in this network.')

        input = tf.keras.layers.Input(shape=(self.input_size, 1))

        x = self._initial_block(input, 1)

        # Add residual blocks
        k = 1
        subsample = False
        pad_channels = False
        for res_id in range(1, NUM_RESIDUAL_BLOCKS + 1):
            if res_id == 1:
                x = self._residual_block_type_1(x, k)
            else:
                x = self._residual_block_type_2(x, k, subsample, pad_channels)

            # The convolutional layers (initial_filters)*k filters, where k starts
            # out as 1 and is incremented every 4-th residual block
            if (res_id % 4) == 0:
                k += 1
                pad_channels = True
            else:
                pad_channels = False

            # Every alternate residual block subsamples its inputs
            subsample = res_id % 2 == 0

        y = self._final_block(x, self.num_classes)

        model = tf.keras.Model(input, y)

        return model

    def visualise_models(self, args):
        """
        Generates plots for each of the constructed models showcasing their architecture.

        This method utilises TensorFlow's plot_model functionality to visualise the architecture
        of each model maintained in the models_dict. The resulting plots provide an overview
        of layers, their connectivity, and the shape of the data flowing through the model.

        https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
        """

        for model_name, model in self.models_dict.items():

            path_to_save = os.path.join(args[model_name][f'{model_name}_plot'], f'{model_name}_plot.png')

            tf.keras.utils.plot_model(
                model,
                to_file=path_to_save,
                show_shapes=True,
                show_dtype=True,
                show_layer_names=True,
                rankdir='LR',  # creates a horizontal plot.
                expand_nested=True,
                dpi=300,
                layer_range=None,
                show_layer_activations=True,
                show_trainable=False
            )

            print(f'\nModel {model_name} visualised and saved as {model_name}.png')
