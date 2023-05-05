# Heart Failure Prediction using RNNs: A Reproducibility Study
This repository contains the code and documentation for a reproduction study that aims to validate the findings of an experiment examining the potential benefits of employing deep learning techniques, particularly recurrent neural networks (RNNs), to improve the prediction of heart failure onset by capturing the temporal relationships among events documented in electronic health records (EHRs). The primary objective is to compare the performance of RNN models with conventional methods that do not consider temporality, aiming to enhance early detection and ultimately improve patient outcomes.

# Scope of Reproducibility
The scope of reproducibility for this experiment involves:
* Re-implementing the original study's methodology
* Validating the results by comparing the performance of RNN models, which utilize gated recurrent units (GRUs), against conventional methods such as regularized logistic regression, multilayer perceptron (MLP), support vector machine (SVM), and K-nearest neighbor (KNN) classifiers
* Investigating the enhanced model performance in predicting initial diagnosis of heart failure within a 12- to 18-month observation window by leveraging temporal relationships among events in EHRs

# Addressed Claims from the Original Paper
We aim to address the following claims from the original paper:

* Improved Prediction Performance: The use of RNN models with GRUs leads to better performance in predicting the initial diagnosis of heart failure within a 12- to 18-month observation window compared to conventional methods such as regularized logistic regression, MLP, SVM, and KNN classifiers.
* Temporal Information Utilization: The incorporation of temporal relationships among events in EHRs through RNN models provides a significant advantage in predicting heart failure onset over conventional methods that do not account for temporality.

# Data
This reproduction study uses the MIMIC-III Clinical Database as the primary data source. To gain access to the dataset, follow the instructions provided here.

Once you have access to the MIMIC-III dataset, download and store the data files in the data/ directory within the repository.

# Getting Started
* Prerequisites
* Python 3.7 or later
* TensorFlow 2.6 or later
* scikit-learn 0.24 or later
* pandas 1.3 or later
* NumPy 1.21 or later
