
# /----------------------[ Input Arguments ]----------------------/
ARG_SEARCH_DIRECTORY = 'search_directory'
ARG_TRAINING_SET_FPATH = 'training_set_fpath'
ARG_TEST_SET_FPATH = 'test_set_fpath'
INITIAL_WEIGHTS = "initial_weights"
SELECT_DATASET = 'mitbih'
MODELS_DIR = 'models'
TRAINING_DIR = "training"
EVALUATIONS_DIR = "evaluations"
PLOTS_DIR = 'plot_model'


# /----------------------[ Load Data ]----------------------/
TEST_SIZE = 0.3
PADDING_MAXLEN = 200

# /----------------------[ Resampling ]----------------------/
SEED = 42
KEY_SIMPLE_RESAMPLING = 'simple'
KEY_SMOTE_RESAMPLING = 'smote'
KEY_ADASYN_RESAMPLING = 'adasyn'
TRAINING_SUFFIX = '_train'
EVALUATION_SUFFIX = '_eval'
OVERSAMPLING_TECHNIQUES = [KEY_SIMPLE_RESAMPLING, KEY_SMOTE_RESAMPLING, KEY_ADASYN_RESAMPLING]

# /----------------------[ Build Models ]----------------------/
INITIAL_NUM_FILTERS = 16  # Conv1D param as per the paper (default 64)
NUM_RESIDUAL_BLOCKS = 4  # Total number of residual blocks as per the paper (default 16)
KERNEL_SIZE = 4  # Conv1D param as per the paper (default 16)
DROPOUT_RATE = 0.2  # We have assumed it since this is not specified in the paper (default 0.5)
UNITS_DENSE = 128
KEY_BASELINE_NN = 'baseline_neural_network'
KEY_DEEP_CNN = 'deep_cnn'
MODELS = [KEY_BASELINE_NN, KEY_DEEP_CNN]
COMPILE_MODELS_BOOL = True

# /----------------------[ Train Models ]----------------------/
NUM_EPOCHS = 100
BATCH_SIZE = 2048  # to ensure that each batch has a decent chance of containing all the classes.
PATIENCE_EARLY_STOPPING = 15
PATIENCE_REDUCE_LR = 10
