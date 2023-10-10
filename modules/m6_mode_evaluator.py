
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple, List, Any
from modules.m4_model_builder import ModelArchitectureBuilder
from modules.default import (
    MODELS, OVERSAMPLING_TECHNIQUES,
    TRAINING_SUFFIX, EVALUATION_SUFFIX
)


class ModelEvaluator:
    """
    Class for evaluating machine learning models on test data.

    This class provides methods to load trained models, predict on test data,
    generate classification reports, plot confusion matrices, and visualise various metrics.
    Designed to support multi-class classification problems and offers visualisation
    tools like ROC curves.

    Attributes:
    - args: Dictionary containing paths and other evaluation-related arguments.
    - X_test: Test feature data.
    - y_test_encoded: One-hot encoded test labels.
    - num_classes: The number of unique classes in the dataset.
    - trained_models_dict: A dictionary holding loaded models for evaluation.

    Methods:
    - load_models: Loads pre-trained models for evaluation.
    - _predict: Predicts using a given model and returns predicted probabilities and labels.
    - _generate_report: Creates and saves a classification report for predictions.
    - _plot_data: General function to plot either confusion matrices or other metrics.
    - _plot_roc_curve: Plots the Receiver Operating Characteristic (ROC) curve for each class.
    - evaluate_models: Orchestrates the evaluation process using all the methods.
    """

    def __init__(self, args: Dict[str, Any],
                 X_test: np.ndarray,
                 y_test_encoded: np.ndarray,
                 num_classes: int):
        """
        Initialises the ModelEvaluator.

        Parameters:
        - args (Dict[str, Any]): Dictionary containing paths and other arguments.
        - X_test (np.ndarray): Test features.
        - y_test_encoded (np.ndarray): One-hot encoded test labels.
        - num_classes (int): Number of classes in the dataset.
        """
        self.args = args
        self.X_test = X_test
        self.y_test_encoded = y_test_encoded
        self.num_classes = num_classes
        self.trained_models_dict = {}

    def load_models(self) -> None:
        """
        Loads all models for evaluation.
        """
        custom_metrics = {f'f1_class_{i}': ModelArchitectureBuilder._f1_score_per_class(i)
                          for i in range(self.num_classes)}
        for model_name in MODELS:
            for oversampling_technique in OVERSAMPLING_TECHNIQUES:
                trained_model_path = os.path.join(self.args[model_name][f'{oversampling_technique}{TRAINING_SUFFIX}'],
                                                  f'best_{model_name}_{oversampling_technique}.h5')
                if os.path.exists(trained_model_path):
                    loaded_model = tf.keras.models.load_model(trained_model_path, custom_objects=custom_metrics)
                    self.trained_models_dict[(model_name, oversampling_technique)] = loaded_model
                else:
                    raise FileNotFoundError(f"Model at path {trained_model_path} not found!")

    def _predict(self, model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts using a trained model.

        Parameters:
        - model (tf.keras.Model): Trained model.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Tuple of predicted and true labels.
        """
        y_pred_probs = model.predict(self.X_test)
        y_pred_labels = np.argmax(y_pred_probs, axis=-1)
        y_true_labels = np.argmax(self.y_test_encoded, axis=-1)

        # Check for new classes in test set
        unique_test_labels = np.unique(y_true_labels)
        if max(unique_test_labels) >= self.num_classes:
            raise ValueError("Test set contains classes not seen during training.")

        return y_pred_labels, y_true_labels

    def _generate_report(self, y_pred: np.ndarray,
                         y_true: np.ndarray,
                         model_name: str,
                         oversampling_technique: str) -> Dict[str, Any]:
        """
        Generates and saves classification report.

        Parameters:
        - y_pred (np.ndarray): Predicted labels.
        - y_true (np.ndarray): True labels.
        - model_name (str): Name of the model.
        - oversampling_technique (str): Oversampling technique used.

        Returns:
        - Dict[str, Any]: Classification report.
        """

        labels = [i for i in range(self.num_classes)]  # should be integers
        report = classification_report(y_true, y_pred, output_dict=True, labels=labels)
        renamed_report = {('class_' + k if k.isdigit() else k): v for k, v in report.items()}
        renamed_report_df = pd.DataFrame(renamed_report).T
        path_to_save = os.path.join(self.args[model_name][f'{oversampling_technique}{EVALUATION_SUFFIX}'],
                                    f'classification_report_{model_name}_{oversampling_technique}.csv')
        renamed_report_df.to_csv(path_to_save)

        return renamed_report

    def _save_confusion_matrix_csv(self, matrix: np.ndarray, matrix_type: str, model_name: str,
                                   oversampling_technique: str) -> None:
        """
        Saves the confusion matrix (counts or percentages) to a CSV file.

        Parameters:
        - matrix (np.ndarray): Confusion matrix.
        - matrix_type (str): Type of the matrix ('counts' or 'percentages').
        - model_name (str): Name of the model.
        - oversampling_technique (str): Oversampling technique used.
        """
        df = pd.DataFrame(matrix, columns=[f"class_{i}" for i in range(self.num_classes)],
                          index=[f"class_{i}" for i in range(self.num_classes)])
        path_to_save = os.path.join(self.args[model_name][f'{oversampling_technique}{EVALUATION_SUFFIX}'],
                                    f'confusion_matrix_{matrix_type}_{model_name}_{oversampling_technique}.csv')
        df.to_csv(path_to_save)

    def _plot_confusion_matrix(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: List[str],
                               model_name: str,
                               oversampling_technique: str) -> None:
        """
        Plots and saves the confusion matrix for the given true and predicted labels.
        """
        matrix = confusion_matrix(y_true, y_pred)

        plt.figure()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Counts) | {model_name} | {oversampling_technique}')
        path_to_save = os.path.join(self.args[model_name][f'{oversampling_technique}{EVALUATION_SUFFIX}'],
                                    f'confusion_matrix_counts_{model_name}_{oversampling_technique}.png')
        plt.tight_layout()
        plt.savefig(path_to_save)
        plt.close()

        # Save CSVs for counts
        self._save_confusion_matrix_csv(matrix, 'counts', model_name, oversampling_technique)

    def _plot_metrics(self,
                      class_names: List[str],
                      model_name: str,
                      oversampling_technique: str,
                      report: Dict[str, Any] = None) -> None:
        """
        Plots precision, recall, and F1-score metrics for each class.

        This method visualises the precision, recall, and F1-score metrics as bar graphs
        for each class in the given classification report. The resulting plot is saved to
        a designated path.

        Parameters:
        - class_names (List[str]): List of class names to be plotted on the x-axis.
        - model_name (str): Name of the evaluated model used in the title and filename of the plot.
        - oversampling_technique (str): Name of the oversampling technique used in the title and filename.
        - report (Dict[str, Any]): Classification report dictionary containing metrics for each class.

        Raises:
        - ValueError: If the `report` is not provided.
        """

        if report is None:
            raise ValueError("Report is required for metrics plotting.")

        # This part stays the same as in your original `plot_metrics_bar` method
        precision = []
        recall = []
        f1_score = []

        for class_name in class_names:
            metrics = report.get(str(class_name))
            precision.append(metrics.get('precision', 0))
            recall.append(metrics.get('recall', 0))
            f1_score.append(metrics.get('f1-score', 0))

        x = np.arange(len(class_names))
        width = 0.2
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1_score, width, label='F1 Score')
        plt.xlabel('Class Name')
        plt.ylabel('Scores')
        plt.title(f'Scores by class | {model_name} | {oversampling_technique}')
        plt.xticks(x, class_names)
        plt.legend()
        path_to_save = os.path.join(self.args[model_name][f'{oversampling_technique}{EVALUATION_SUFFIX}'],
                                    f'metrics_{model_name}_{oversampling_technique}.png')

        plt.tight_layout()
        plt.savefig(path_to_save)
        plt.close()

    def evaluate_models(self) -> None:
        """
        Evaluates all trained models.
        """
        if not self.trained_models_dict:
            raise ValueError("No pre-trained models loaded. Please use load_models() to load models first.")

        for (model_name, oversampling_technique), model in self.trained_models_dict.items():
            print(f"\nEvaluating {model_name} model with {oversampling_technique} oversampling...")

            # 1. Predict using the model
            y_pred, y_true = self._predict(model)

            # 2. Generate and save a classification report
            report = self._generate_report(y_pred, y_true, model_name, oversampling_technique)

            # 3. Plot the confusion matrix and metrics
            class_names = [f"class_{i}" for i in range(self.num_classes)]
            self._plot_confusion_matrix(y_true, y_pred, class_names, model_name, oversampling_technique)
            self._plot_metrics(class_names, model_name, oversampling_technique, report)
