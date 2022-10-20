[![License mit](https://img.shields.io/badge/license-mit-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
# High Fidelity FaceShift Model Based on Attention 
## Installation

Clone this repo.
```bash
git clone https://github.com/Friedrich-M/AttnSwap.git
cd AttnSwap/
```

## Dataset Preparation
High Resolution Dataset [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ)
```
mkdir vggface2_crop_arcfacealign_224
```
Put the downloaded dataset under this folder

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

Download arcface_checkpoint.tar from SimSwap and put it under the arcface_model folder
```
mkdir arcface_model
cd arcface_model
tar xvf arcface_checkpoint.tar
cd ../
```

Download vgg_normalised.pth from the SAnet and put it under the vggface2_crop_arcfacealign_224 folder
```
mkdir vggface2_crop_arcfacealign_224
```

## Training New Models
```
sh train_attnswap.sh
```

There are many options you can specify. Please use `python train_attnswap.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

To log training, use `--use_tensorboard` for Tensorboard. The logs are stored at `[checkpoints]/[name]/`.
