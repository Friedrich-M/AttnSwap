[![License mit](https://img.shields.io/badge/license-mit-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# High Fidelity FaceShift Model Based on Attention 
<img width="700" alt="image" src="https://user-images.githubusercontent.com/85838942/197096449-66fb5bad-39d1-4e59-a096-0934be14e446.jpg">

## Installation

Clone this repo.
```bash
git clone https://github.com/Friedrich-M/AttnSwap.git
cd AttnSwap/
```

## Dataset Preparation
High Resolution Dataset Celeba-256
```
mkdir data_psp
```
Put the downloaded dataset under this folder

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

Download **CurricularFace_Backbone.pth** and put it under the arcface_psp_model folder
```
mkdir arcface_psp_model
```

Download psp_ffhq_encode.pt from the PSP and put it under the psp_model folder
```
mkdir psp_model
```

## Training New Models
```
sh train_attnswap.sh
```

There are many options you can specify. Please use `python train_attnswap.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

To log training, use `--use_tensorboard` for Tensorboard. The logs are stored at `[checkpoints]/[name]/`.

## Current progress
![step_41000](https://user-images.githubusercontent.com/85838942/197097112-4b2fd716-7ec5-4b31-a6fe-e67d1d08bfa2.jpg)



