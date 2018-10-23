# Domain adaptive segmentation

This code implements the methods that were described in the paper: 

Joris Roels, Julian Hennies, Yvan Saeys, Wilfried Philips, Anna Kreshuk, "Domain adaptive segmentation in volume electron microscopy imaging", ISBI 2019 (submitted). 

We provide three classification domain adaptation (DA) approaches ([DANN](https://arxiv.org/abs/1502.02791), [CORAL](https://arxiv.org/abs/1607.01719) and [MMD](https://arxiv.org/abs/1505.07818)) for encoder-decoder segmentation. Additionally, we provide a new domain adaptation technique (called Y-Net) which introduces a reconstruction decoder to incorporate relevant target-specific features. 

We acknowledge the code of [DANN](https://github.com/fungtion/DANN), [CORAL](https://github.com/SSARCandy/DeepCORAL) and [MMD](https://github.com/OctoberChang/MMD-GAN), which was used in this work. 

## Requirements
- Tested with Python 3.5
- Python libraries: 
    - torch 0.4
    - tensorboardX (for TensorBoard usage)
    - tifffile (for data loading)
    - [imgaug](https://github.com/aleju/imgaug) (data augmentation) 
- [EPFL](https://cvlab.epfl.ch/data/data-em/) data should be in the [data/epfl](data/epfl) folder for testing the demo script. 

## Instructions
The proposed DA approaches consist of two steps: 1) unsupervised domain alignment with source labels and 2) supervised finetuning with available target labels. 
1) Unsupervised domain alignment: 
    `python train_unsupervised.py --method ynet --target drosophila`
2) Supervised finetuning with 10% of the target labels: 
    `python train_supervised.py --method ynet --target drosophila --frac_target_labels 0.1`
