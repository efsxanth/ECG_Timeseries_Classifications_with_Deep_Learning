
import numpy as np
from modules.default import PADDING_MAXLEN
from modules.m1_arguments_and_configuration import ArgumentManager
from modules.m2_dataset_manager import DatasetManager
from modules.m3_resampling import MultiClassOversampler
from modules.m4_model_builder import ModelArchitectureBuilder
from modules.m5_model_trainer import ModelTrainer
from modules.m6_mode_evaluator import ModelEvaluator

if __name__ == '__main__':
    """
    =====================
         ENTRY POINT
    =====================

    Entry Point for the Multi-class Classification Pipeline

    This script acts as the primary interface for the end-to-end multi-class classification pipeline. 
    It facilitates:
    1. Argument parsing for file paths and other relevant parameters.
    2. Data loading and preprocessing using `DatasetManager`.
    3. Data resampling to handle imbalanced datasets with `MultiClassOversampler`.
    4. Neural network model architecture creation for various configurations with `ModelArchitectureBuilder`.
    5. Training the specified models using the resampled training data and validation data with `ModelTrainer`.
    6. Evaluating the trained models using `ModelEvaluator` on test data. This evaluation includes performance
       metrics and visual tools.
    
    The workflow is presented in a step-by-step manner, ensuring clarity in the order of operations and easy debugging.
    
    Example:
        $ python <script_name>.py --search_directory /path/to/data
    
    Modules Used:
        ArgumentManager, DatasetManager, MultiClassOversampler, ModelArchitectureBuilder, ModelTrainer, ModelEvaluator
    
    Note:
        Ensure all necessary module dependencies are installed and that paths provided in arguments exist.
    """

    print('\nArguments...')
    args_manager = ArgumentManager()
    args = args_manager.filepath_arguments

    print('\nLoading data...')
    data_loader = DatasetManager(args=args)
    num_classes = len(np.unique(data_loader.y_train))

    print('\nResampling...')
    resampling = MultiClassOversampler(X_train=data_loader.X_train, y_train=data_loader.y_train)
    resampling.oversample()

    data_loader.unload_training_data()  # unload X_train and y_train

    print('\nBuilding models...')
    builder = ModelArchitectureBuilder(input_size=PADDING_MAXLEN, num_classes=num_classes)
    builder.build_all()
    builder.visualise_models(args=args)
    model_architectures_dict = builder.models_dict

    print('\nTraining models...')
    trainer = ModelTrainer(args=args,
                           X_train_resampled_dict=resampling.X_train_resampled_dict,
                           y_train_resampled_encoded_dict=resampling.y_train_resampled_encoded_dict,
                           X_val=data_loader.X_val,
                           y_val_encoded=data_loader.y_val_encoded,
                           model_architectures_dict=model_architectures_dict)
    trainer.training_process()

    data_loader.unload_validation_data()
    resampling.unload_resampled_data()

    print('\nEvaluating models...')
    model_evaluator = ModelEvaluator(args=args, X_test=data_loader.X_test,
                                     y_test_encoded=data_loader.y_test_encoded, num_classes=num_classes)
    model_evaluator.load_models()
    model_evaluator.evaluate_models()

    print('\nDone!')
    