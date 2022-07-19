# Food101

## Dataset :

The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101k images. The labels for the test images have been manually cleaned, while the training set contains some noise.

## Goal

In the DeepFood: "Deep Learning-based Food Image Recognition for Computer-aided Dietary Assessment" paper that presented the datasets the scientist achieved an top 1 -accuracy of 76.3%.
My was to surpass this accuracy implementing various state of the arts models, also implementing various image augmentation, methods to prevent overfitting & fine tuning methods.

## Description :

The aim of this project is to achieve a better accuracy on the Food101 dataset than the one described in the orginal paper.

## How to :

I will use transfer learning with an Efficientnet backbone, first as a feature extractor and then fine tuning by unfreezing some layers.
I implemented a basic Baseline model, ResNET50, EfficientNetB0-4 model and a Vision Transformer model

## Framework used :

All models will be created using TensorFlow and Keras.

## Deployment :

At the end of the experience, I will use Docker to make my most performant model, and be able to test it on various image.

## Results :
