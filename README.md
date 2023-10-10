# ECG_Timeseries_Classifications_with_Deep_Learning

## Overview

Electrocardiograms (ECGs) are a fundamental diagnostic tool in cardiology, with millions of recordings made each year. The detection of irregular heart rhythms or arrhythmias from these recordings is essential for diagnosing various heart-related diseases. This project leverages the power of Deep Learning (DL) to delve deep into this challenge. Specifically, the purpose of the project is to investigate the effectiveness of three oversampling techniques — Simple oversampling (duplicating random samples from minority classes), SMOTE, and ADASYN — in enhancing the performance of DL models on ECG classification tasks.

## Contents

- Dataset
- Pipeline Architecture
- Results
- References

## Dataset

The data is sourced from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?datasetId=29414&language=Python) and is based on the Physionet's MIT-BIH Arrhythmia Dataset, widely used for research in ECG classification.

- Number of Training samples: 87,553
- Number of Test samples: 21,892
- Time-series ECG data: Columns 0 to 186
- Class labels: Column 187
- Number of Classes (Categories): 5

## Pipeline Architecture

The project comprises several modules:
1. Arguments & Configuration - Manages argument parsing and configuration for machine learning processes.
2. Data Manager - Manages datasets for loading, padding, splitting, and provides access to train, validation, and test sets.
3. Resampling - Offers functionality for oversampling multi-class datasets using various techniques.
4. Model Builder - Constructs and manages model architectures for classification tasks.
5. Model Trainer - Trains machine learning models and visualizes performance metrics.
6. Model Evaluator - Evaluates pre-trained models on test data and generates insightful metrics.

## Results

The results indicate that:
- Deep CNN models consistently outperform the Baseline Neural Networks.
- The Simple oversampling technique often provides the best balance between precision and recall compared to SMOTE and ADASYN.
- The results align with those of leading research in the field, such as Rajpurkar, P., et al., 2017.

## References

1. Rajpurkar, P., Hannun, A. Y., Haghpanahi, M., Bourn, C., & Ng, A. Y. (2017). Cardiologist-level arrhythmia detection with convolutional neural networks. arXiv preprint [arXiv:1707.01836](https://arxiv.org/pdf/1707.01836.pdf).
2. Fazeli, Shayan. ECG Heartbeat Categorization Dataset. [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?datasetId=29414&language=Python), (access: 19th September 2023)
3. Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG Heartbeat Classification: A Deep Transferable Representation. arXiv preprint [arXiv:1805.00794](https://arxiv.org/pdf/1805.00794.pdf).
