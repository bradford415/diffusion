# Diffusion
Repository of diffusion models mostly for learning purposes.

## Table of Contents
* [Training a Model](#training-a-model)
* [Resources](#resources)

## Latent Diffusion



## DDPM

### Training
`python scripts/train.py scripts/configs/train-cifar10-pc.yaml scripts/configs/models/ddpm.yaml`

__cifar-10 dataset__

`python scripts/train.py scripts/configs/train-cifar10-pc.yaml scripts/configs/models/ddpm.yaml`

## DDPM results
### cifar-10

* results after 400k steps, batch size 64, image size 32x32

![generated_images](https://github.com/user-attachments/assets/7d140815-10a2-43cf-8731-3f7bd94dd2ca)



## Resources
### Denoising Diffusion Probabilistic Model (DDPM)
* [DDPM Paper](https://arxiv.org/abs/2006.11239)
* [lucidrains DDPM PyTorch Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [Another PyTorch Implementation](https://github.com/w86763777/pytorch-ddpm/tree/master)
  * This one is written slightly different and for some parts is easier to follow
* [CVPR DDPM Tutorial](https://cvpr2023-tutorial-diffusion-models.github.io/)
  * Powerpoint explains the equations very well and helped me link the code to the equations
* [Annotated Unet](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)
  * Very good annotated Unet code and other great annotations and explanations of Stable Diffusion
