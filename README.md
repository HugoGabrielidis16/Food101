# Food101

## Dataset :

The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101k images. The labels for the test images have been manually cleaned, while the training set contains some noise.

## Goal

In the DeepFood: "Deep Learning-based Food Image Recognition for Computer-aided Dietary Assessment" paper that presented the datasets the scientist achieved an top 1 -accuracy of 76.3%.
My was to surpass this accuracy implementing various state of the arts models, also implementing various image augmentation, methods to prevent overfitting & fine tuning methods.

# Description of the model used

- In order to establish a baseline I used a simple ConvNET architecture.
- A ResNET50 models used both as a freeze feature extractor and a model to FineTuned.
- EfficientNETB0-B4 also used either as a feature extractor or a model to FineTuned.
- A Vision Transformer implemented from scratch.

## How to :

I will use transfer learning with an Efficientnet backbone, first as a feature extractor and then fine tuning by unfreezing some layers.
I implemented a basic Baseline model, ResNET50, EfficientNetB0-4 model and a Vision Transformer model

## Stack used :

matplotlib==3.5.2  <br />
numpy==1.22.4  <br />
scikit_learn==1.1.1  <br />
tensorflow==2.8.2  <br />
tensorflow_datasets==4.6.0 <br />
wandb==0.12.21

## Deployment :

At the end of the experience, I will use Docker to make my most performant model, and be able to test it on various image.

## Results :

- Baseline :
- ResNET50 feature extractor :
- ResNET50 FineTuned :
- EfficientNetB0 feature extractor :
- EfficientNetB0 FineTuned :
- EfficientNetB1 feature extractor :
- EfficientNetB1 FineTuned :
- EfficientNetB2 feature extractor :
- EfficientNetB2 FineTuned :
- EfficientNetB3 feature extractor :
- EfficientNetB3 FineTuned :
- EfficientNetB4 feature extractor :
- EfficientNetB4 FineTuned :
