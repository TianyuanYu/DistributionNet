# Robust Person Re-identification by Modelling Feature Uncertainty

This repo contains the reference source code for the paper [Robust Person Re-identification by Modelling Feature Uncertainty](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Robust_Person_Re-Identification_by_Modelling_Feature_Uncertainty_ICCV_2019_paper.pdf) in ICCV 2019. In this project, we provide Resnet-Baseline and DistributionNet architectures and results with 10% random noise on Market dataset.


## Enviroment
 - Python 2.7
 - Tensorflow 1.3.0

## Getting started

Download dataset from [here](https://drive.google.com/drive/folders/1VUpNKRjaxOh3A_sbgsWdKuhq7BOHOOC9?usp=sharing)

Folder 'Market' includes original Market dataset (training, query, and gallery) and 10% random noise Market training dataset file. 

Folder 'result' includes trained models of Resnet-Baseline and DistributionNet

Folder 'pretrained_model' includes Resnet-50 pretrained in ImageNet

## Train and Test
Run
```bash pipeline.sh```

## Results
The test results will be recorded in `./result/model_name/rank.txt`
