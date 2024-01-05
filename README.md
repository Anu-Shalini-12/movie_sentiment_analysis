
# Sentiment Analysis Model

This repository contains code for training and testing a sentiment analysis model on the IMDB dataset. The model is implemented using both a Naive Bayes classifier and a Neural Network using TensorFlow and Keras.

## Overview

The project involves training and evaluating two types of models for sentiment analysis on the IMDB dataset - a Naive Bayes classifier and a Neural Network. The Naive Bayes model uses the CountVectorizer for text representation, while the Neural Network utilizes an Embedding layer followed by an LSTM layer.

## Requirements

Make sure you have the following libraries installed:

- pandas
- scikit-learn
- TensorFlow
- matplotlib (for visualization)

You can install these dependencies using the following command:

```bash
pip install pandas scikit-learn tensorflow matplotlib

##Train the Models
-Run the NLP_Train.py script to train both the Naive Bayes and Neural Network models.

Test the Models
-Run the NLP_Test.py script to load the trained Neural Network model and evaluate it on the test set.

## File Descriptions

- **`NLP_Train.py`**: This script is responsible for training both the Naive Bayes and Neural Network models. It loads the IMDB dataset, preprocesses the data, and then uses the data to train the models. The Naive Bayes model is implemented using the CountVectorizer, while the Neural Network uses an Embedding layer followed by an LSTM layer. The trained Neural Network model is saved as `sentiment_model.h5`.

- **`NLP_Test.py`**: This script is used to test the trained Neural Network model. It loads the saved model, tokenizes the test data, and evaluates the model on the test set. The script prints the accuracy and classification report for the model's performance on the test data.

- **`sentiment_model.h5`**: This file contains the saved Neural Network model in HDF5 format. It is generated during the training process and loaded during testing.

##Author
-Anu-shalini-12


