# Locality Alignment

[![arXiv](https://img.shields.io/badge/arXiv-2410.11087-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2410.11087)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Timm](https://img.shields.io/badge/TIMM-1.0.8-black.svg?style=for-the-badge&logo=huggingface)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/github/license/iancovert/locality-alignment?style=for-the-badge)](LICENSE)

This is a repository for *locality alignment*, a post-training stage that helps vision transformers (ViTs) learn to extract local image semantics (i.e., class contents for each image region). We use an efficient fine-tuning procedure to teach this capability using only self-supervision — masked embedding self-consistency (MaskEmbed). These techniques are introduced in [this paper](https://arxiv.org/abs/2410.11087).

Locality alignment is useful for pre-trained backbones like CLIP and SigLIP that are only exposed to image-level supervision (e.g., image-caption pairs) instead of dense, region-level supervision. Our paper shows improved performance on a local feature probing benchmark, and better benchmark performance in vision-language models. See our repositories below for usage in these tasks:

- [Probing benchmark](https://github.com/iancovert/patch-seg)
- [Vision-language models](https://github.com/iancovert/prismatic-vlms)


# Installation and usage

First, clone the repository and install it in your conda environment:

```bash
git clone https://github.com/iancovert/locality-alignment.git
cd locality-alignment
pip install -e .
```

Training utilities are provided in the `locality_alignment` package, our model configs are in `configs`, and the main training script is in `scripts/train.py`.

**Data.** Before training you first need to set up your training dataset, for example ImageNet1k or ImageNet21k. If these are already downloaded on your machine, you can symlink them at `data/imagenet` and `data/imagenet21k`. Otherwise, you can download each dataset using the standard approach (see [here](https://github.com/pytorch/examples/tree/main/imagenet) for ImageNet1k and [here](https://arxiv.org/abs/2104.10972) for ImageNet21k).

**Launching training.** Jobs should be launched from the root directory of the repository, and the easiest way to specify hyperparameters is with a config file. To launch a training run with the CLIP ViT-B backbone on 4 GPUs, you can use the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 scripts/train.py --config configs/clip-vit-b.yaml
```

To train on a single GPU, you can launch directly with Python:

```bash
python scripts/train.py --config configs/clip-vit-b.yaml
```

Training progress is logged to wandb, and checkpoints are saved in the `output` directory.

# How it works

The key idea behind MaskEmbed is that pre-trained models have latent knowledge of local semantics that we can extract using masking. As a source of self-supervision, we mask out patches and use the frozen model to produce a masked view, which only reflects the unmasked contents. We then fork the model into a trainable copy accompanied by a lightweight decoder, and use these to predict the masked view: the fine-tuned encoder predicts rich patch embeddings given the unmasked image, which are masked and then passed to the decoder to predict the masked view. The model is trained to minimize the difference between these two predictions.

<p align="center">
  <img src="training_diagram.png" alt="MaskEmbed" style="max-width: 540px; width: 100%;" />
</p>


### Acknowledgement

We thank Ross Wightman for creating and maintaining the [timm](https://github.com/huggingface/pytorch-image-models) repository. Our training script is a modified version of timm's [train.py](https://github.com/huggingface/pytorch-image-models/blob/main/train.py), and we use timm for loading all pre-trained models and slightly modifying the ViT architecture.


### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2410.11087):

```bibtex
@article{covert2024locality,
  title = {Locality Alignment Improves Vision-Language Models},
  author = {Covert, Ian and Sun, Tony and Zou, James and Hashimoto, Tatsunori},
  year = {2024},
  journal = {arXiv preprint arXiv:2410.11087},
}
```
