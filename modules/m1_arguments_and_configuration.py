
import argparse
import glob
import os
from typing import Dict, Optional
from modules.default import (
    ARG_SEARCH_DIRECTORY,
    SELECT_DATASET,
    MODELS_DIR, TRAINING_DIR, EVALUATIONS_DIR, PLOTS_DIR,
    TRAINING_SUFFIX, EVALUATION_SUFFIX,
    INITIAL_WEIGHTS,
    MODELS,
    OVERSAMPLING_TECHNIQUES,
    ARG_TRAINING_SET_FPATH,
    ARG_TEST_SET_FPATH,
)


class ArgumentManager:
    """
        Manages and constructs essential file paths based on user arguments and predefined naming conventions.

        The class is responsible for:
        1. Parsing command-line arguments to determine the data directory.
        2. Scanning for training and test datasets within the specified directory.
        3. Constructing subdirectories for each model, their oversampling techniques, and corresponding evaluations.
        4. Creating a dedicated subdirectory for the initial weights of each model for training.
        5. Providing an organized dictionary structure that encapsulates all the generated paths.

        Attributes:
        - filepath_arguments (dict): A dictionary containing all the constructed paths.
    """

    def __init__(self, args: Optional[Dict[str, str]] = None):

        if not args:
            args = self._parse_arguments()

        self._filepath_arguments = self._define_fpath_arguments(args_dict=args)

    @property
    def filepath_arguments(self):
        return self._filepath_arguments

    def __repr__(self):
        return f"<ArgumentManager with paths: {self._filepath_arguments}>"

    def __str__(self):
        return f"ArgumentManager containing paths: {self._filepath_arguments}"

    @staticmethod
    def _parse_arguments():
        """
        Parse command-line arguments to determine the data directory.

        Returns:
            dict: Parsed command-line arguments.

        Raises:
            ValueError: If no arguments are provided.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument(ARG_SEARCH_DIRECTORY, type=str,
                            help='Pass the search directory containing the data of interest.')
        args = parser.parse_args()

        if not args:  # Add error handling for no arguments
            raise ValueError("No arguments provided.")

        return vars(args)

    def _create_model_subdirs(self, base_dir, models_folder, model):
        """
        Create and return subdirectories for a given model and all its oversampling techniques.

        Parameters:
            base_dir (str): Base directory path.
            models_folder (str): Models directory name.
            model (str): Name of the model for which directories need to be created.

        Returns:
            dict: Paths of all subdirectories created for the model.
        """

        oversampling_paths = {
            f'{oversampling_technique}{TRAINING_SUFFIX}': self._create_dir(base_dir, models_folder, model,
                                                                           TRAINING_DIR, oversampling_technique)
            for oversampling_technique in OVERSAMPLING_TECHNIQUES
        }

        # Add evaluation directories for each oversampling technique
        evaluation_paths = {
            f'{oversampling_technique}{EVALUATION_SUFFIX}': self._create_dir(base_dir, models_folder, model,
                                                                             EVALUATIONS_DIR, oversampling_technique)
            for oversampling_technique in OVERSAMPLING_TECHNIQUES
        }

        # Add a directory for plots for the model
        plot_paths = {f'{model}_plot': self._create_dir(base_dir, models_folder, model, PLOTS_DIR)}

        return {**oversampling_paths, **evaluation_paths, **plot_paths}

    @staticmethod
    def _create_dir(*path_parts):
        """
        Create and return a directory path using the provided parts.

        Parameters:
            *path_parts: Variable number of string arguments representing parts of the directory path.

        Returns:
            str: Full directory path created from the provided parts.
        """

        dir_path = os.path.join(*path_parts)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def _define_fpath_arguments(self, args_dict: dict) -> dict:
        """
        Defines essential file paths based on provided arguments and established naming conventions.

        The method achieves the following tasks:
        1. Scans for training and test datasets within the provided directory.
        2. If either dataset is not located, a FileNotFoundError is raised.
        3. Constructs subdirectories for every model, corresponding oversampling techniques, and their evaluation
           directories in the model directory.
        4. Further creates a dedicated subdirectory for the initial weights of each model for training.
        5. Creates a dedicated subdirectory for the plots of each model.
        6. Augments the given dictionary with paths for training data, test data, model subdirectories, evaluation
           directories, initial weights, and model plots.

        Parameters:
        - args_dict (dict): Contains the search directory specified by the user.

        Returns:
        - dict: The enhanced dictionary with crucial file paths for training data, test data, model subdirectories,
                evaluation directories, initial weights, and model plots.

        Raises:
        - FileNotFoundError: If the training or test datasets aren't located in the defined directory.
        - Exception: For any other generic anomalies encountered during file path resolution or directory creation.

        Example:
        Given the user input '/path/to/data' as the search directory, the method might return:
        {
            'search_directory': '/path/to/data',
            'train_data_path': '/path/to/data/train.csv',
            'test_data_path': '/path/to/data/test.csv',
            'model1': {
                'oversampling_technique1_train': '/path/to/data/models/model1/training/oversampling_technique1',
                'oversampling_technique1_eval': '/path/to/data/models/model1/evaluations/oversampling_technique1',
                'oversampling_technique2_train': '/path/to/data/models/model1/training/oversampling_technique2',
                'oversampling_technique2_eval': '/path/to/data/models/model1/evaluations/oversampling_technique2',
                ...
                'initial_weights_train': '/path/to/data/models/model1/training/initial_weights',
                'model1_plot': '/path/to/data/models/model1/plot_model'
            },
            'model2': {...},
            ...
        }
        """

        try:
            list_fpaths = glob.glob(os.path.join(args_dict[ARG_SEARCH_DIRECTORY], '**', '*.csv'), recursive=True)

            # Extract training and test set paths.
            training_set_path = next((fp for fp in list_fpaths if SELECT_DATASET in fp and '_train.' in fp), "")
            test_set_path = next((fp for fp in list_fpaths if SELECT_DATASET in fp and '_test.' in fp), "")

            if not training_set_path:
                raise FileNotFoundError(
                    f"No training dataset found in the specified directory: {args_dict[ARG_SEARCH_DIRECTORY]}")

            if not test_set_path:
                raise FileNotFoundError(
                    f"No test dataset found in the specified directory: {args_dict[ARG_SEARCH_DIRECTORY]}")

            # Create subdirectories for each model, oversampling technique,
            # evaluation directories, and initial weights directory
            model_paths = {
                model: {
                    **self._create_model_subdirs(args_dict[ARG_SEARCH_DIRECTORY], MODELS_DIR, model),
                    f'{INITIAL_WEIGHTS}{TRAINING_SUFFIX}': self._create_dir(args_dict[ARG_SEARCH_DIRECTORY],
                                                                            MODELS_DIR, model, TRAINING_DIR,
                                                                            INITIAL_WEIGHTS)
                }
                for model in MODELS
            }

            args_dict.update(model_paths)
            args_dict[ARG_TRAINING_SET_FPATH] = training_set_path
            args_dict[ARG_TEST_SET_FPATH] = test_set_path

        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}. The file was not found.")
            raise

        except Exception as generic_error:
            print(f"An error occurred: {generic_error}")
            raise

        return args_dict
