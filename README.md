# Extreme-Learning-Machine

This repository provides a comprehensive implementation of Extreme Learning Machines (ELMs) for image classification alongside a logistic regression baseline using polynomial feature expansions. The code explores advanced techniques including mixup data augmentation, ensemble modeling, and a direct least-squares solution for the fully connected layer.

---

## Repository Overview

The repository is organized into two main components, each in its own folder:

### 1. Logistic Regression with Polynomial Feature Expansion (Folder: `task1`)

These implementations study the effect of polynomial feature expansion on logistic regression performance. They are used as a baseline and include:

- **Polynomial Feature Computation:**  
  - In `task1a.py`, features are grouped by polynomial degree (constant, linear, quadratic, cubic) and weighted by learnable gating parameters.  
  - In `task1.py`, polynomial features (constant, linear, squares, cross terms, cubes, and mixed terms) are concatenated into one tensor.
  
- **Model and Training:**  
  Both files generate synthetic datasets using a fixed underlying weight pattern and train logistic regression models via stochastic gradient descent (SGD) with options for binary cross-entropy or RMSE loss.

---

### 2. Extreme Learning Machines (ELMs) for Image Classification (Folder: `task2`)

The ELM implementations focus on image classification on the CIFAR-10 dataset using a two-stage model:
  
- **Fixed Convolutional Feature Extractor:**  
  The `MyExtremeLearningMachine` class uses a fixed convolutional layer (initialized with Gaussian weights and then frozen) for feature extraction.

- **Trainable Fully Connected Layer:**  
  - In `task2.py`, the fully connected (FC) layer is trained using SGD.
  - In `task2a.py`, a direct least-squares solver is used to optimize the FC layer.  
   
- **Mixup Data Augmentation:**  
  The `MyMixUp` class implements mixup augmentation, which blends images and their one-hot encoded labels to create augmented samples that help regularize the model.

- **Ensemble Techniques:**  
  The `MyEnsembleELM` class creates an ensemble of several independently initialized ELM models. The ensemble prediction is obtained by averaging individual outputs to reduce variance.

- **Additional Features:**  
  In `task2a.py`, a random hyperparameter search routine is provided to optimize parameters such as the number of hidden maps, kernel size, and weight initialization standard deviation. Both files include helper functions for evaluation (accuracy, macro F1-score) and visualization (montages of predictions and mixup-augmented images).

---

## Installation and Requirements

Ensure you have Python 3.x installed along with the following dependencies:

- PyTorch  
- Torchvision  
- NumPy  
- Pillow

Install the required packages via pip:

```bash
pip install torch torchvision numpy pillow
