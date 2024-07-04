# deep-learning-challenge

# Alphabet Soup Charity Funding Predictor

This repository contains a machine learning project that predicts the success of charity funding applications using deep learning techniques. The goal is to assist the nonprofit foundation Alphabet Soup in selecting the applicants for funding with the best chance of success in their ventures.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Optimization](#model-optimization)
- [Dependencies](#dependencies)


## Overview

The project aims to build and optimize a deep learning model to predict whether applicants funded by Alphabet Soup will be successful. The dataset contains metadata about each organization, such as application type, affiliated sector, use case for funding, and other relevant features.

## Project Structure

alphabet-soup-charity/

├── models/

│ ├── AlphabetSoupCharity.h5

│ ├── AlphabetSoupCharity_Optimization.h5

│ ├── AlphabetSoupCharity_Optimization_2.h5

│ └── AlphabetSoupCharity_Optimization_3.h5

├── notebooks/

│ ├── AlphabetSoupCharity.ipynb

│ ├── AlphabetSoupCharity_Optimization.ipynb

│ ├── AlphabetSoupCharity_Optimization_2.ipynb

│ └── AlphabetSoupCharity_Optimization_3.ipynb

├── README.md
└── 

## Data Preprocessing

1. **Loading Data**: The dataset is loaded from `charity_data.csv`.
2. **Dropping Irrelevant Columns**: The `EIN` and `NAME` columns are dropped.
3. **Encoding Categorical Variables**: Categorical variables are encoded using `pd.get_dummies()`.
4. **Handling Rare Categories**: Rare categories in `APPLICATION_TYPE` and `CLASSIFICATION` are grouped into an "Other" category.
5. **Splitting Data**: The data is split into features (`X`) and target (`y`), and then into training and testing sets.
6. **Scaling Data**: The features are scaled using `StandardScaler`.

## Model Building

The initial model is built using TensorFlow and Keras with the following structure:

- Input layer with the number of features
- First hidden layer with 80 neurons and ReLU activation
- Second hidden layer with 30 neurons and ReLU activation
- Output layer with 1 neuron and sigmoid activation

The model is compiled with the Adam optimizer, binary crossentropy loss, and accuracy as the metric. It is trained for 100 epochs with a batch size of 32.

## Model Optimization

Three optimization attempts were made:

1. **Increased Neurons**: Increased the number of neurons in the hidden layers.
2. **Added Hidden Layer**: Added a third hidden layer.
3. **Changed Activation Functions and Increased Epochs**: Changed activation functions to `tanh` and increased the number of epochs to 200.

Each optimized model is saved as an HDF5 file.

## Dependencies

- pandas
= numpy
- scikit-learn
- tensorflow
- matplotlib
