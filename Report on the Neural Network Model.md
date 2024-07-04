# Report on the Neural Network Model

## Overview of the Analysis

The purpose of this analysis is to create a binary classifier using deep learning techniques to predict whether applicants funded by Alphabet Soup will be successful in their ventures. The analysis involves preprocessing the data, building and training a neural network model, evaluating its performance, and optimizing the model to achieve a target accuracy.

## Results

### Data Preprocessing

- Target Variable: The target variable for the model is IS_SUCCESSFUL, which indicates whether the funding was used effectively.
- Feature Variables: The feature variables are all the columns except EIN and NAME. These include:

  - APPLICATION_TYPE
  - AFFILIATION
  - CLASSIFICATION
  - USE_CASE
  - ORGANIZATION
  - STATUS
  - INCOME_AMT
  - SPECIAL_CONSIDERATIONS
  - ASK_AMT

- Removed Variables: The EIN and NAME columns were removed from the input data as they are identification columns and do not contribute to the prediction.

## Compiling, Training, and Evaluating the Model

### Neurons, Layers, and Activation Functions:

 - Initial Model: The initial model had two hidden layers. The first hidden layer had 80 neurons with ReLU activation, and the second hidden layer had 30 neurons with ReLU activation. The output layer had 1 neuron with sigmoid activation.
 
 - Optimization Attempts:
   1) Attempt 1: Increased the number of neurons to 100 and 50 in the hidden layers, respectively.
   2) Attempt 2: Added a third hidden layer with 30 neurons.
   3) Attempt 3: Changed activation functions to tanh and increased the number of epochs to 200.

### Model Performance:

 - Initial Model Performance: The initial model achieved an accuracy of around 72%.
   1) Attempt 1 Performance: The model with increased neurons achieved an accuracy of around 73%.
   2) Attempt 2 Performance: The model with an additional hidden layer achieved an accuracy of around 74%.
   3) Attempt 3 Performance: The model with tanh activation and increased epochs achieved an accuracy of around 75%.

### Steps to Increase Model Performance:

 - Adjusted the number of neurons in the hidden layers.
 - Added more hidden layers to the model.
 - Changed the activation functions to tanh.
 - Increased the number of epochs during training.

## Summary
 - Overall Results: The deep learning model was optimized through multiple attempts, achieving a maximum accuracy of around 75% with the third attempt.
 - Recommendation: To further improve the model's performance, consider using other machine learning algorithms such as Random Forest, Gradient Boosting, or Support Vector Machines, which might handle the data more effectively and provide better accuracy. Additionally, hyperparameter tuning and cross-validation can help in identifying the best model configuration.