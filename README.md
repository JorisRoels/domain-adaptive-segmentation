# Domain adaptive segmentation

This code implements the methods that were described in the paper: 

Joris Roels, Julian Hennies, Yvan Saeys, Wilfried Philips, Anna Kreshuk, ["Domain adaptive segmentation in volume electron microscopy imaging"](https://arxiv.org/abs/1810.09734), ISBI 2019 (accepted). 

We provide three classification domain adaptation (DA) approaches ([domain adversarial training (DAT)](https://arxiv.org/abs/1502.02791), [deep correlation alignment (CORAL)](https://arxiv.org/abs/1607.01719) and [maximum mean discrepancy (MMD)](https://arxiv.org/abs/1505.07818)) for encoder-decoder segmentation. Additionally, we provide a new domain adaptation technique (called Y-Net) which introduces a reconstruction decoder to incorporate relevant target-specific features. 

We acknowledge the code of [DAT](https://github.com/fungtion/DANN), [CORAL](https://github.com/SSARCandy/DeepCORAL) and [MMD](https://github.com/OctoberChang/MMD-GAN), which was used in this work. 

## Requirements
- Tested with Python 3.6
- Required Python libraries (these can be installed with `pip install -r requirements.txt`): 
    - numpy
    - tifffile
    - scipy
    - scikit-image
    - imgaug
    - torch
    - torchvision
    - jupyter (optional)
    - progressbar2 (optional)
    - tensorboardX (optional, for tensorboard usage)
    - tensorflow (optional, for tensorboard usage)

## Requirements
- Required data: 
  - [EPFL mitochondria dataset (source)](https://cvlab.epfl.ch/data/data-em/)
  - [VNC mitochondria dataset (target)](https://github.com/unidesigner/groundtruth-drosophila-vnc)

## Usage
We provide a [notebook](ynet.ipynb) that illustrates usage of our method. Note that the data path might be different, depending on where you downloaded the data. 
