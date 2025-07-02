# Product Attribute detection based on Images - Amazon ML Challenge Solution

This repository presents a comprehensive solution for the Amazon Machine Learning Challenge, focusing on accurately extracting numerical values and their corresponding units for various product attributes (such as item weight, volume, voltage, etc.) from product listings, leveraging both image and textual data. The project employs a robust pipeline encompassing sophisticated data preprocessing, a multi-modal deep learning architecture, and a streamlined prediction workflow.

## Key Methodologies and Components

The solution's core lies in its structured approach to data handling and an advanced deep learning model designed for multi-task prediction.

### 1\. Data Preprocessing

The initial phase involves meticulous preparation of the raw data (`train.csv` and `test.csv`). A custom parsing logic is applied to the `entity_value` column to robustly extract numerical values and their associated units, handling diverse formats including ranges and bracketed entries. A crucial step is the standardization of units; various weight (e.g., grams, kilograms, ounces), volume (e.g., millilitres, litres, gallons), and power (e.g., horsepower, watts) units are converted into a consistent base unit to ensure data uniformity for model training. This includes careful outlier filtering based on established numerical value ranges.

### 2\. Model Architecture

The deep learning model is engineered to handle both image and structured data inputs and perform two distinct prediction tasks simultaneously:

  * **Vision Transformer (ViT) for Image Features:** The `ViTFeatureExtractor` and `ViTModel` from the `transformers` library are utilized to extract high-dimensional, contextual features from product images. This enables the model to leverage visual information pertinent to product attributes.
  * **Multi-Task Prediction Heads:**
      * **Regression Head:** A dedicated Convolutional Neural Network (CNN) based regressor is designed to predict the precise numerical value of the attribute.
      * **Classification Head:** Another CNN-based classifier is responsible for predicting the correct unit (e.g., 'kilogram', 'litre', 'watt') associated with the predicted value.
  * **PyTorch Framework:** The entire model architecture and training pipeline are built using PyTorch, offering flexibility in defining custom layers and efficient computation.

### 3\. Training Process

The model training was conducted entirely on **free Google Colab T4 GPUs**, leveraging their computational power for efficient deep learning.

The training process involves:

  * **Data Pipelining:** Custom `Dataset` and `DataLoader` classes are used to efficiently feed batches of processed data to the model, including image tensors and numerical/categorical features.
  * **Feature Scaling:** `StandardScaler` is applied to numerical features, and `LabelEncoder` is used for unit categories to prepare them for neural network consumption.
  * **Loss Functions:** A combination of appropriate loss functions for both tasks (e.g., Mean Squared Error for regression and Cross-Entropy Loss for classification) is optimized during training.
  * **Optimization:** An optimizer (e.g., Adam or SGD with momentum, depending on the final configuration in the notebook) updates the model weights based on the computed gradients from the combined loss.
  * **Epoch-based Training:** The model is trained over multiple epochs, with performance monitored through metrics like Mean Squared Error for regression and accuracy/F1-score for classification.

### 4\. Prediction and Output Generation

Upon successful training, the final models are used to generate predictions on the unseen test dataset:

  * **Model Loading:** The pre-trained ViT model, CNN regressor, and CNN classifier are loaded into memory.
  * **Test Data Inference:** The test dataset undergoes the same preprocessing and feature extraction steps as the training data to ensure consistency.
  * **Batch Prediction:** Predictions for both numerical values and units are generated for the entire test set in batches for efficiency.
  * **Inverse Transformation:** Predicted numerical values are inverse-transformed from their scaled representation to their original scale, and predicted unit indices are converted back to human-readable unit labels.
  * **Submission File Generation:** The final predicted numerical values and their corresponding units are compiled and saved into `predicted_values_final.csv`, formatted as required for the challenge submission. A utility `predict_single_image` function is also included to demonstrate real-time inference for individual product images.

## Repository Structure (Brief)

  * `Amazon.ipynb`: Contains code for data loading, preprocessing, `entity_value` parsing, and unit conversions.
  * `FullDataTrainingFinal.ipynb`: Implements the deep learning model architecture (ViT + CNNs), training loop, and evaluation metrics.
  * `FinalTest.ipynb`: Handles model loading, inference on the test set, and generation of the final submission file.

## Future Enhancements

  * Exploring more advanced regularization techniques to prevent overfitting.
  * Implementing alternative unit conversion strategies to improve robustness for edge cases.
  * Investigating different backbone architectures for image feature extraction beyond ViT.
  * Detailed hyperparameter tuning using automated tools for optimal model performance.
