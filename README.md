# Food101

## Dataset :

The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101k images. The labels for the test images have been manually cleaned, while the training set contains some noise.

## Goal

In the DeepFood: "Deep Learning-based Food Image Recognition for Computer-aided Dietary Assessment" paper that presented the datasets the scientist achieved a top 1 -accuracy of 76.3%.
My aim was to surpass this accuracy implementing various state of the arts models, also implementing fine tuning methods & various methods to prevent overfitting such as image augmentation.

# Description of the model used

- In order to establish a baseline I used a simple ConvNET architecture.
- A ResNET50 models used both as a freeze feature extractor and a model to be finetuned.
- EfficientNETB0-B4 also used either as a feature extractor or a model to be finetuned.
- A Vision Transformer implemented from scratch.

## How to use

To verify my results you can either clone the git folder and use the bash with the command line :
chmod +x test_script.sh
script.sh "model_name"

## Stack used :

matplotlib==3.5.2 <br />
numpy==1.22.4 <br />
scikit_learn==1.1.1 <br />
tensorflow==2.8.2 <br />
tensorflow_datasets==4.6.0 <br />
wandb==0.12.21

## ToDo

Use a validation split to fine tuned my model ( can't fine tuned on test data )

## Results :

Tested on 7000 samples

- Baseline : accuracy - 0.1739 | precision - | recall - | F1 score
- ResNET50 feature extractor :
- ResNET50 fineTuned :
- EfficientNetB0 feature extractor : accuracy - 0.71 | precision - 0.72 | recall - 0.72 | F1 score - 0.71
- EfficientNetB0 fineTuned : accuracy - 0.80 | precision - 0.81 | recall - 0.80 | F1 score - 0.80
- EfficientNetB1 feature extractor : accuracy - 0.68 | precision - 0.69 | recall - 0.68 | F1 score - 0.68
- EfficientNetB1 fineTuned : accuracy - 0.81| precision - 0.83 | recall - 0.82 | F1 score - 0.82
- EfficientNetB2 feature extractor : accuracy - 0.64 | precision - 0.64 | recall - 0.64 | F1 score - 0.63
- EfficientNetB2 fineTuned : accuracy - 0.83 | precision - 0.84 | recall - 0.83| F1 score -0.83a
- EfficientNetB3 feature extractor : accuracy - 0.64 | precision - 0.65 | recall - 0.64 | F1 score - 0.63
- EfficientNetB3 fineTuned : accuracy - 0.84| precision - 0.84 | recall - 0.83 | F1 score - 0.83
- EfficientNetB4 feature extractor :
- EfficientNetB4 fineTuned :

Best model :
To achieve a best model I will finetune the hyper parameters, on the finetuned EfficientNetB3 : I will look for the best LR, and if this doesn't allow me go beyound 0.90 accuracy I will also change the number of neurons in the last layers
