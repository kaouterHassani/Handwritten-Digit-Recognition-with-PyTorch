# Handwritten-Digit-Recognition-with-PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch to recognize handwritten digits from the MNIST dataset. The model is trained to classify images of digits (0-9) and evaluated on the test dataset.

## Table Of Contents
* [Overview](#Overview)
* [Dataset](#Dataset)
* [Model Architecture](#Model_Architecture) 
* [Setup and Installation](#Setup_and_Installation)
* [Training and Evaluation](#Training_and_Evaluation)
* [Results](#Results)

## Overview
The goal of this project is to build a CNN model that classifies handwritten digits from the MNIST dataset, which contains 60,000 training images and 10,000 test images of grayscale digits (0-9). The project includes training the model, testing its accuracy, and visualizing the correct and incorrect predictions.

## Dataset
The [MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) is used for training and evaluating the model. It consists of 28x28 grayscale images of digits, with 60,000 samples for training and 10,000 for testing.

   * **Training dataset**: 60,000 images
   * **Test dataset**: 10,000 images

## Model Architecture
The CNN architecture is composed of two convolutional layers followed by max-pooling layers and dropout. It ends with two fully connected layers.

   *  **Conv Layer 1:** 1 input channel (grayscale), 10 output channels, 5x5 kernel
   * **Conv Layer 2:** 10 input channels, 20 output channels, 5x5 kernel, followed by a dropout layer
   * **Fully Connected Layer 1:** 320 input features, 50 output features
   * **Fully Connected Layer 2:** 50 input features, 10 output features (one for each class 0-9)

The activation functions used are `ReLU`, and `softmax` is used at the end to get class probabilities.

## Setup and Installation
* Python 3.8+
* PyTorch 
* torchvision
* Matplotlib

## Training and Evaluation
* **Training Process:** The model is trained using the `Adam optimizer` with a `learning rate` of `0.001`. The loss function used is `CrossEntropyLoss`, and during training, the model's parameters are updated using `backpropagation`.

* **Evaluation:** The model is evaluated using the `test set`. The `accuracy` is calculated by comparing the model's predictions with the actual labels.

## Results 
The model achieved the following performance on the MNIST test set:

 * **Test set:** average loss: 0.0001
 * **Accuracy:** 9723/10000 (97%)
    
You can Check the incorrect predictions made by the model [here]()
