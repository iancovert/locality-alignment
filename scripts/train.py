"""Modified from timm training script: https://github.com/huggingface/pytorch-image-models/blob/main/train.py"""

import os
import json
import time
import yaml
import argparse
import importlib
import logging
from pathlib import Path
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as DDP

from timm import utils
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    Mixup,
    FastCollateMixup,
    AugMixDataset,
)
from timm.models import (
    create_model,
    safe_model_name,
    resume_checkpoint,
    model_parameters,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler as GradScaler

from locality_alignment import (
    load_checkpoint_auto,
    convert_to_teacher_model,
    convert_to_student_model,
    CheckpointSaver,
    MaskEmbedDecoder,
    MaskEmbedStudent,
    BernoulliMaskSampler,
    BlockwiseMaskSampler,
    UniformMaskSampler,
)

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

has_compile = hasattr(torch, "compile")

_logger = logging.getLogger("train")

# The first arg parser parses only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below.
# fmt: off
config_parser = parser = argparse.ArgumentParser(description="Training Config", add_help=False)
parser.add_argument("-c", "--config", default="", type=str, metavar="FILE", help="YAML config file specifying default arguments")
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Dataset parameters.
group = parser.add_argument_group("Dataset parameters")
parser.add_argument("--data-dir", metavar="DIR", help="path to dataset (root dir)")
parser.add_argument("--dataset", metavar="NAME", default="", help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument("--train-split", metavar="NAME", default="train", help="dataset train split (default: train)")
group.add_argument("--val-split", metavar="NAME", default="validation", help="dataset validation split (default: validation)")
parser.add_argument("--train-num-samples", default=None, type=int, metavar="N", help="Manually specify num samples in train split, for IterableDatasets.")
parser.add_argument("--val-num-samples", default=None, type=int, metavar="N", help="Manually specify num samples in validation split, for IterableDatasets.")
group.add_argument("--dataset-download", action="store_true", default=False, help="Allow download of dataset for torch/ and tfds/ datasets that support it.")
group.add_argument("--class-map", default="", type=str, metavar="FILENAME", help='path to class to idx mapping file (default: "")')
group.add_argument("--input-img-mode", default=None, type=str, help="Dataset image conversion mode for input images.")
group.add_argument("--input-key", default=None, type=str, help="Dataset key for input images.")
group.add_argument("--target-key", default=None, type=str, help="Dataset key for target labels.")

# Model parameters.
group = parser.add_argument_group("Model parameters")
group.add_argument("--student-model", default="vit_base_patch16_224.augreg_in21k_ft_in1k", type=str, metavar="MODEL", help='Name of model to train (default: "vit_base_patch16_224.augreg_in21k_ft_in1k")')
group.add_argument("--student-head", default="linear", type=str, help='type of output head (default: "linear" or "mlp")')
group.add_argument("--student-pretrained", type=bool, default=True, help="Start with pretrained version of specified network (if avail)")
group.add_argument("--student-checkpoint", default="", type=str, metavar="PATH", help="Load this checkpoint into student model before student conversion (default: none)")
group.add_argument("--freeze-student", type=bool, default=False, help="Freeze the student encoder model")
group.add_argument("--teacher-model", default="vit_base_patch16_224.augreg_in21k_ft_in1k", type=str, metavar="MODEL", help='Name of model to use for supervision (default: "vit_base_patch16_224.augreg_in21k_ft_in1k")')
group.add_argument("--teacher-pretrained", type=bool, default=True, help="Start with pretrained version of specified network (if avail)")
group.add_argument("--teacher-checkpoint", default="", type=str, metavar="PATH", help="Load this checkpoint into teacher model (default: none)")
group.add_argument("--skip-blocks", default=1, type=int, metavar="MODEL", help="Number of blocks to skip for teacher's internal representation (default: 1)")
group.add_argument("--prefix-only", type=bool, default=False, metavar="MODEL", help="Whether to reconstruct the prefix tokens only (default: False)")
group.add_argument("--loss-function", type=str, default="mse", metavar="MODEL", help='Loss function for reconstruction, either "mse" or "cosine" (default: "mse")')
group.add_argument("--decoder-layers", default=2, type=int, metavar="MODEL", help="Number of transformer blocks for decoder (default: 2)")
group.add_argument("--decoder-width", default=None, type=int, metavar="MODEL", help="Decoder embedding dimension (default: 4)")
group.add_argument("--decoder-num-heads", default=None, type=int, metavar="MODEL", help="Number of self-attention heads for decoder (default: inherited from student)")
group.add_argument("--decoder-qk-norm", default=None, type=bool, metavar="MODEL", help="Whether to use qk_norm in decoder (default: inherited from student)")
group.add_argument("--decoder-qkv-bias", default=None, type=bool, metavar="MODEL", help="Whether to use qkv_bias in decoder (default: inherited from student)")
group.add_argument("--decoder-init-values", default=1e-6, type=float, metavar="MODEL", help="Decoder LayerScale initialization (default: 1e-6)")
group.add_argument("--resume", default="", type=str, metavar="PATH", help="Resume full model and optimizer state from checkpoint (default: none)")
group.add_argument("--no-resume-opt", action="store_true", default=False, help="prevent resume of optimizer state when resuming model")
group.add_argument("--img-size", type=int, default=None, metavar="N", help="Image size (default: None => model default)")
group.add_argument("--in-chans", type=int, default=None, metavar="N", help="Image input channels (default: None => 3)")
group.add_argument("--input-size", default=None, nargs=3, type=int, metavar="N N N", help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty")
group.add_argument("--crop-pct", default=None, type=float, metavar="N", help="Input image center crop percent (for validation only)")
group.add_argument("--mean", type=float, nargs="+", default=None, metavar="MEAN", help="Override mean pixel value of dataset")
group.add_argument("--std", type=float, nargs="+", default=None, metavar="STD", help="Override std deviation of dataset")
group.add_argument("--interpolation", default="", type=str, metavar="NAME", help="Image resize interpolation type (overrides model)")
group.add_argument("-b", "--batch-size", type=int, default=128, metavar="N", help="Input batch size for training (default: 128)")
group.add_argument("-vb", "--validation-batch-size", type=int, default=None, metavar="N", help="Validation batch size override (default: None)")
group.add_argument("--channels-last", action="store_true", default=False, help="Use channels_last memory layout")
group.add_argument("--grad-accum-steps", type=int, default=1, metavar="N", help="The number of steps to accumulate gradients (default: 1)")
group.add_argument("--grad-checkpointing", action="store_true", default=False, help="Enable gradient checkpointing through model blocks/stages")
group.add_argument("--model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument("--student-model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)
group.add_argument("--teacher-model-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Scripting / codegen.
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument("--torchscript", dest="torchscript", action="store_true", help="torch.jit.script the full model")
scripting_group.add_argument("--torchcompile", nargs="?", type=str, default=None, const="inductor", help="Enable compilation w/ specified backend (default: inductor).")

# Device and distributed.
group = parser.add_argument_group("Device parameters")
group.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
group.add_argument("--amp", action="store_true", default=False, help="use PyTorch native AMP for mixed precision training")
group.add_argument("--amp-dtype", default="float16", type=str, help="lower precision AMP dtype (default: float16)")
group.add_argument("--no-ddp-bb", action="store_true", default=False, help="Force broadcast buffers for native DDP to off.")
group.add_argument("--synchronize-step", action="store_true", default=False, help="torch.cuda.synchronize() end of each step")
group.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--device-modules", default=None, type=str, nargs="+", help="Python imports for device backend modules.")

# Optimizer parameters.
group = parser.add_argument_group("Optimizer parameters")
group.add_argument("--opt", default="sgd", type=str, metavar="OPTIMIZER", help='Optimizer (default: "sgd")')
group.add_argument("--opt-eps", default=None, type=float, metavar="EPSILON", help="Optimizer Epsilon (default: None, use opt default)")
group.add_argument("--opt-betas", default=None, type=float, nargs="+", metavar="BETA", help="Optimizer Betas (default: None, use opt default)")
group.add_argument("--momentum", type=float, default=0.9, metavar="M", help="Optimizer momentum (default: 0.9)")
group.add_argument("--weight-decay", type=float, default=0.01, help="weight decay (default: 0.1)")
group.add_argument("--clip-grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)")
group.add_argument("--clip-mode", type=str, default="norm", help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument("--layer-decay", type=float, default=None, help="layer-wise learning rate decay (default: None)")
group.add_argument("--opt-kwargs", nargs="*", default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters.
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument("--sched", type=str, default="cosine", metavar="SCHEDULER", help='LR scheduler (default: "step"')
group.add_argument("--sched-on-updates", action="store_true", default=False, help="Apply LR scheduler step on update instead of epoch end.")
group.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate, overrides lr-base if set (default: None)")
group.add_argument("--lr-base", type=float, default=0.1, metavar="LR", help="base learning rate: lr = lr_base * global_batch_size / base_size")
group.add_argument("--lr-base-size", type=int, default=256, metavar="DIV", help="base learning rate batch size (divisor, default: 256).")
group.add_argument("--lr-base-scale", type=str, default="", metavar="SCALE", help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct", help="learning rate noise on/off epoch percentages")
group.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT", help="learning rate noise limit percent (default: 0.67)")
group.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV", help="learning rate noise std-dev (default: 1.0)")
group.add_argument("--lr-cycle-mul", type=float, default=1.0, metavar="MULT", help="learning rate cycle len multiplier (default: 1.0)")
group.add_argument("--lr-cycle-decay", type=float, default=0.5, metavar="MULT", help="amount to decay each learning rate cycle (default: 0.5)")
group.add_argument("--lr-cycle-limit", type=int, default=1, metavar="N", help="learning rate cycle limit, cycles enabled if > 1")
group.add_argument("--lr-k-decay", type=float, default=1.0, help="learning rate k-decay for cosine/poly (default: 1.0)")
group.add_argument("--warmup-lr", type=float, default=1e-5, metavar="LR", help="warmup learning rate (default: 1e-5)")
group.add_argument("--min-lr", type=float, default=0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0 (default: 0)")
group.add_argument("--epochs", type=int, default=300, metavar="N", help="number of epochs to train (default: 300)")
group.add_argument("--epoch-repeats", type=float, default=0.0, metavar="N", help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).")
group.add_argument("--start-epoch", default=None, type=int, metavar="N", help="manual epoch number (useful on restarts)")
group.add_argument("--decay-milestones", default=[90, 180, 270], type=int, nargs="+", metavar="MILESTONES", help="list of decay epoch indices for multistep lr. must be increasing")
group.add_argument("--decay-epochs", type=float, default=90, metavar="N", help="epoch interval to decay LR")
group.add_argument("--warmup-epochs", type=int, default=5, metavar="N", help="epochs to warmup LR, if scheduler supports")
group.add_argument("--warmup-prefix", action="store_true", default=False, help="Exclude warmup period from decay schedule.")
group.add_argument("--cooldown-epochs", type=int, default=0, metavar="N", help="epochs to cooldown LR at min_lr, after cyclic schedule ends")
group.add_argument("--patience-epochs", type=int, default=10, metavar="N", help="patience epochs for Plateau LR scheduler (default: 10)")
group.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE", help="LR decay rate (default: 0.1)")

# Augmentation and regularization parameters.
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument("--no-aug", action="store_true", default=False, help="Disable all training augmentation, override other train aug args")
group.add_argument("--train-crop-mode", type=str, default=None, help="Crop-mode in train")
group.add_argument("--scale", type=float, nargs="+", default=[0.08, 1.0], metavar="PCT", help="Random resize scale (default: 0.08 1.0)")
group.add_argument("--ratio", type=float, nargs="+", default=[3.0 / 4.0, 4.0 / 3.0], metavar="RATIO", help="Random resize aspect ratio (default: 0.75 1.33)")
group.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip training aug probability")
group.add_argument("--vflip", type=float, default=0.0, help="Vertical flip training aug probability")
group.add_argument("--color-jitter", type=float, default=0.4, metavar="PCT", help="Color jitter factor (default: 0.4)")
group.add_argument("--color-jitter-prob", type=float, default=None, metavar="PCT", help="Probability of applying any color jitter.")
group.add_argument("--grayscale-prob", type=float, default=None, metavar="PCT", help="Probability of applying random grayscale conversion.")
group.add_argument("--gaussian-blur-prob", type=float, default=None, metavar="PCT", help="Probability of applying gaussian blur.")
group.add_argument("--aa", type=str, default=None, metavar="NAME", help='Use AutoAugment policy. "v0" or "original". (default: None)')
group.add_argument("--aug-repeats", type=float, default=0, help="Number of augmentation repetitions (distributed training only) (default: 0)")
group.add_argument("--aug-splits", type=int, default=0, help="Number of augmentation splits (default: 0, valid: 0 or >=2)")
group.add_argument("--reprob", type=float, default=0.0, metavar="PCT", help="Random erase prob (default: 0.)")
group.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
group.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
group.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")
group.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, mixup enabled if > 0. (default: 0.)")
group.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha, cutmix enabled if > 0. (default: 0.)")
group.add_argument("--cutmix-minmax", type=float, nargs="+", default=None, help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)")
group.add_argument("--mixup-prob", type=float, default=1.0, help="Probability of performing mixup or cutmix when either/both is enabled")
group.add_argument("--mixup-switch-prob", type=float, default=0.5, help="Probability of switching to cutmix when both mixup and cutmix enabled")
group.add_argument("--mixup-mode", type=str, default="batch", help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument("--mixup-off-epoch", default=0, type=int, metavar="N", help="Turn off mixup after this epoch, disabled if 0 (default: 0)")
group.add_argument("--train-interpolation", type=str, default="random", help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")
group.add_argument("--drop-path", type=float, default=None, metavar="PCT", help="Drop path rate (default: None)")
group.add_argument("--drop-block", type=float, default=None, metavar="PCT", help="Drop block rate (default: None)")

# Model exponential moving average (EMA) parameters.
group = parser.add_argument_group("Model exponential moving average parameters")
group.add_argument("--model-ema", action="store_true", default=False, help="Enable tracking moving average of model weights.")
group.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.")
group.add_argument("--model-ema-decay", type=float, default=0.9998, help="Decay factor for model weights moving average (default: 0.9998)")
group.add_argument("--model-ema-warmup", action="store_true", help="Enable warmup for model EMA decay.")

# Miscellaneous.
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
group.add_argument("--worker-seeding", type=str, default="all", help="worker seed mode (default: all)")
group.add_argument("--log-interval", type=int, default=100, metavar="N", help="how many batches to wait before logging training status")
group.add_argument("--recovery-interval", type=int, default=0, metavar="N", help="how many batches to wait before writing recovery checkpoint")
group.add_argument("--checkpoint-hist", type=int, default=10, metavar="N", help="number of checkpoints to keep (default: 10)")
group.add_argument("--checkpoint-retain-interval", type=int, default=50, metavar="N", help="interval to save checkpoints regardless of metric (default: 50)")
group.add_argument("-j", "--workers", type=int, default=4, metavar="N", help="how many training processes to use (default: 4)")
group.add_argument("--save-images", action="store_true", default=False, help="save images of input bathes every log interval for debugging")
group.add_argument("--pin-mem", action="store_true", default=False, help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
group.add_argument("--no-prefetcher", action="store_true", default=False, help="disable fast prefetcher")
group.add_argument("--output", default="", type=str, metavar="PATH", help="path to output folder (default: none, current dir)")
group.add_argument("--experiment", default="", type=str, metavar="NAME", help="name of train experiment, name of sub-folder for output")
group.add_argument("--project", default=None, type=str, metavar="NAME", help="name of wandb project")
group.add_argument("--wandb-id", default=None, type=str, metavar="NAME", help="unique id for wandb run (used for resuming)")
group.add_argument("--log-wandb", action="store_true", default=False, help="log training and validation metrics to wandb")
group.add_argument("--use-multi-epochs-loader", action="store_true", default=False, help="use the multi-epochs-loader to save time at the beginning of every epoch")

# Masking parameters.
group = parser.add_argument_group("Masking parameters")
group.add_argument("--mask-sampling", default="uniform", type=str, help='Mask sampling distribution ("uniform", "bernoulli" or "blockwise")')
group.add_argument("--bernoulli-mask-ratio", default=0.5, type=float, help="Ratio of image to mask under Bernoulli sampling (default: 0.5)")
group.add_argument("--blockwise-mask-ratio", default=0.4, type=float, help="Target ratio of image to mask under block-wise sampling (default: 0.4)")
group.add_argument("--antithetical-sampling", action="store_true", default=True, help="Whether to add the complement for all mask samples")
group.add_argument("--include-null-mask", action="store_true", default=True, help="Whether to add the empty mask for all samples, to reconstruct the teacher's full-image prediction")
# fmt: on


def _parse_args():
    # Check if we have a config file.
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file is specified.
    args = parser.parse_args(remaining)

    # Cache the args as a string to save in the output directory.
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    utils.random_seed(args.seed, args.rank)

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # Verify loss function.
    assert args.loss_function in ("mse", "cosine", "l1", "masked_mse", "unmasked_mse")

    # Create teacher model that allows masking.
    assert args.teacher_pretrained or args.teacher_checkpoint, "Teacher must be pretrained"
    teacher = create_model(
        args.teacher_model,
        pretrained=args.teacher_pretrained,
        in_chans=in_chans,
        num_classes=0,
        scriptable=args.torchscript,
        **args.model_kwargs,
        **args.teacher_model_kwargs,
    )
    if args.teacher_checkpoint:
        non_matching_keys = load_checkpoint_auto(teacher, args.teacher_checkpoint, strict=False)
        if utils.is_primary(args):
            _logger.info(f"Loaded teacher model from {args.teacher_checkpoint}, non-matching keys: {non_matching_keys}")
    teacher_patch_size = teacher.patch_embed.patch_size[0]
    teacher = convert_to_teacher_model(teacher, skip_blocks=args.skip_blocks, prefix_only=args.prefix_only)
    teacher.eval()

    # Log teacher info.
    num_patches = teacher.patch_embed.num_patches
    mask_width = int(num_patches**0.5)
    if utils.is_primary(args):
        _logger.info(
            f"Teacher model num_patches: {num_patches}, mask_width: {mask_width}, patch_size: {teacher_patch_size}"
        )

    # Create student model.
    args.model = args.student_model
    encoder = create_model(
        args.student_model,
        pretrained=args.student_pretrained,
        in_chans=in_chans,
        num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        scriptable=args.torchscript,
        checkpoint_path=args.student_checkpoint,
        **args.model_kwargs,
        **args.student_model_kwargs,
    )
    assert encoder.patch_embed.patch_size[0] == teacher_patch_size
    if args.decoder_width is None:
        args.decoder_width = teacher.embed_dim
    if args.freeze_student:
        encoder.forward = encoder.forward_features
    else:
        encoder = convert_to_student_model(
            encoder,
            output_dim=args.decoder_width,
            output_head=args.student_head,
        )

    # Create decoder model.
    if args.decoder_num_heads is None:
        args.decoder_num_heads = encoder.blocks[0].attn.num_heads
    if args.decoder_qk_norm is None:
        args.decoder_qk_norm = not isinstance(encoder.blocks[0].attn.q_norm, torch.nn.Identity)
    if args.decoder_qkv_bias is None:
        args.decoder_qkv_bias = encoder.blocks[0].attn.qkv.bias is not None

    decoder = MaskEmbedDecoder(
        embed_dim=args.decoder_width,
        num_heads=args.decoder_num_heads,
        num_patch_tokens=num_patches,
        num_prefix_tokens=teacher.num_prefix_tokens,
        output_size=teacher.embed_dim,
        num_layers=args.decoder_layers,
        prefix_only=args.prefix_only,
        qkv_bias=args.decoder_qkv_bias,
        qk_norm=args.decoder_qk_norm,
        init_values=args.decoder_init_values,
        norm_layer=type(encoder.norm),
        act_layer=type(encoder.blocks[0].mlp.act),
        mlp_layer=type(encoder.blocks[0].mlp),
    )

    # Combine backbone and decoder into a single module.
    student = MaskEmbedStudent(encoder, decoder)

    if args.grad_checkpointing:
        student.set_grad_checkpointing(enable=True)

    data_config = resolve_data_config(vars(args), model=teacher, verbose=utils.is_primary(args))

    # Setup augmentation batch splits for split bn.
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # Move model to GPU, enable channels last layout if set.
    student.to(device=device)
    teacher.to(device=device)
    if args.channels_last:
        student.to(memory_format=torch.channels_last)
        teacher.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not args.torchcompile
        student = torch.jit.script(student)
        teacher = torch.jit.script(teacher)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    if args.freeze_student:
        for param in student.encoder.parameters():
            param.requires_grad = False

    optimizer = create_optimizer_v2(
        filter(lambda p: p.requires_grad, student.parameters()),
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # Set up automatic mixed-precision (AMP) loss scaling and op casting.
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        assert args.amp_dtype in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # Fallback to CUDA-only AMP for PyTorch < 1.10.
            assert device.type == "cuda"
            amp_autocast = torch.cuda.amp.autocast
        if device.type == "cuda" and amp_dtype == torch.float16:
            # Loss scaler only used for float16 (half) dtype.
            loss_scaler = GradScaler()
        if utils.is_primary(args):
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            _logger.info("AMP not enabled. Training in float32.")

    # Optionally resume from a checkpoint.
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            student,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # Set up EMA of model weights, SWA could be used here too.
    student_ema = None
    if args.model_ema:
        # Create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper.
        student_ema = utils.ModelEmaV3(
            student,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint_auto(student_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            student_ema = torch.compile(student_ema, backend=args.torchcompile)

    # Set up distributed training.
    if args.distributed:
        student = DDP(student, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA and teacher don't need to be wrapped by DDP.

    if args.torchcompile:
        # Torch compile should be done after DDP.
        assert has_compile, "A version of torch with torch.compile() is required for --compile."
        student = torch.compile(student, backend=args.torchcompile)
        teacher = torch.compile(teacher, backend=args.torchcompile)

    # Set up mask sampler.
    if args.mask_sampling == "uniform":
        mask_sampler = UniformMaskSampler(
            mask_size=num_patches,
            antithetical_sampling=args.antithetical_sampling,
            include_null=args.include_null_mask,
        )
    elif args.mask_sampling == "bernoulli":
        mask_sampler = BernoulliMaskSampler(
            mask_size=num_patches,
            mask_ratio=args.bernoulli_mask_ratio,
            antithetical_sampling=args.antithetical_sampling,
            include_null=args.include_null_mask,
        )
    elif args.mask_sampling == "blockwise":
        mask_sampler = BlockwiseMaskSampler(
            mask_size=num_patches,
            target_ratio=args.blockwise_mask_ratio,
            antithetical_sampling=args.antithetical_sampling,
            include_null=args.include_null_mask,
        )
    else:
        raise ValueError(f"Unknown mask sampling distribution: {args.mask_sampling}")

    # Create the train and eval datasets.
    if args.input_img_mode is None:
        input_img_mode = "RGB" if data_config["input_size"][0] == 3 else "L"
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
    )
    if utils.is_primary(args):
        _logger.info(f"Train dataset length = {len(dataset_train)}")

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
        )
        if utils.is_primary(args):
            _logger.info(f"Val dataset length = {len(dataset_eval)}")

    # Set up mixup / cutmix.
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            num_classes=1000,  # NOTE: arbitrary value, we don't use class labels.
        )
        if args.prefetcher:
            # Collate conflict (need to support deinterleaving in collate mixup).
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # Wrap dataset in AugMix helper.
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # Create data loaders with augmentation pipeline.
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ("tfds" in args.dataset or "wds" in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training.
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config["input_size"],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=eval_workers,
            persistent_workers=eval_workers > 0,
            distributed=args.distributed,
            crop_pct=data_config["crop_pct"],
            pin_memory=args.pin_mem,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # Set up checkpoint saver and eval metric tracking.
    eval_metric = "loss"
    decreasing_metric = eval_metric == "loss"
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.student_model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(args.output if args.output else "./output/train", exp_name)
        saver = CheckpointSaver(
            model=student,
            optimizer=optimizer,
            args=args,
            model_ema=student_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist,
            retain_interval=args.checkpoint_retain_interval,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            if Path(".wandb_token").exists():
                wandb.login(key=Path(".wandb_token").read_text().strip())
            wandb.init(
                project=args.project,
                config=args,
                name=args.experiment,
                id=args.wandb_id,
                resume=args.wandb_id is not None,
            )
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # Set up learning rate schedule and starting epoch.
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # A specified start_epoch will always override the resume epoch.
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
        )

    # Initialize train loader workers (avoids occasional pthread join failures).
    next(iter(loader_train))

    # Scale teacher predictions to have unit variance.
    if args.loss_function == "mse":
        # Get baseline loss.
        baseline_metrics = get_baseline_loss(
            teacher,
            loader_eval,
            mask_sampler,
            args,
            device=device,
            amp_autocast=amp_autocast,
        )

        # Apply normalization.
        scaling_factor = 1 / (baseline_metrics["baseline_loss"] ** 0.5)
        teacher.output_scaling.scale *= scaling_factor
        if utils.is_primary(args):
            _logger.info(
                f"Baseline loss: {baseline_metrics['baseline_loss']:.5f}, "
                f"scaling teacher predictions by factor = {scaling_factor:.2f}"
            )

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                student,
                teacher,
                args.loss_function,
                loader_train,
                optimizer,
                mask_sampler,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                student_ema=student_ema,
                mixup_fn=mixup_fn,
            )

            if loader_eval is not None:
                eval_metrics = validate(
                    student,
                    teacher,
                    args.loss_function,
                    loader_eval,
                    mask_sampler,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                )

                if student_ema is not None and not args.model_ema_force_cpu:
                    ema_eval_metrics = validate(
                        student_ema,
                        teacher,
                        args.loss_function,
                        loader_eval,
                        mask_sampler,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=" (EMA)",
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, "summary.csv"),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # Save proper checkpoint with eval metric.
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # Step LR for next epoch.
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": eval_metrics,
                }
            )

    except KeyboardInterrupt:
        pass

    results = {"all": results}
    if best_metric is not None:
        results["best"] = results["all"][best_epoch - start_epoch]
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))
    print(f"--result\n{json.dumps(results, indent=4)}")


def train_one_epoch(
    epoch,
    student,
    teacher,
    loss_function,
    loader,
    optimizer,
    mask_sampler,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    student_ema=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(student, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    student.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        # FIXME clobbers value for remaining batches and messes up update_idx. Could just increment update_idx?
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # Multiply by accum steps to get equivalent for full update.
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                # Teacher predictions.
                mask = mask_sampler(len(input)).to(input.device)
                mask_ratio = len(mask) // len(input)
                input_repeat = input.repeat_interleave(mask_ratio, 0)
                with torch.no_grad():
                    masked_preds = teacher(input_repeat, mask)

                # Student reconstructions.
                reconstruction = student(input, mask)
                if loss_function == "mse":
                    loss = torch.nn.functional.mse_loss(reconstruction, masked_preds)
                elif loss_function == "cosine":
                    loss = 1 - torch.nn.functional.cosine_similarity(reconstruction, masked_preds, dim=-1).mean()
                elif loss_function == "l1":
                    loss = torch.nn.functional.l1_loss(reconstruction, masked_preds)
                elif loss_function == "masked_mse":
                    # Calculate loss for masked tokens only.
                    if reconstruction.shape[1] > mask.shape[1]:
                        num_prefix_tokens = reconstruction.shape[1] - mask.shape[1]
                        mask = torch.nn.functional.pad(mask, (num_prefix_tokens, 0), value=0)
                    loss = torch.mean((1 - mask).unsqueeze(-1) * (reconstruction - masked_preds) ** 2)
                elif loss_function == "unmasked_mse":
                    # Calculate loss for unmasked tokens only.
                    if reconstruction.shape[1] > mask.shape[1]:
                        num_prefix_tokens = reconstruction.shape[1] - mask.shape[1]
                        mask = torch.nn.functional.pad(mask, (num_prefix_tokens, 0), value=1)
                    loss = torch.mean(mask.unsqueeze(-1) * (reconstruction - masked_preds) ** 2)

            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(student, exclude_head="agc" in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(student, exclude_head="agc" in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with student.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, len(input))
        update_sample_count += len(input)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if student_ema is not None:
            student_ema.update(student, step=num_updates)

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, len(input))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if saver is not None and args.recovery_interval and ((update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # End of for loop.

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    student,
    teacher,
    loss_function,
    loader,
    mask_sampler,
    args,
    device=torch.device("cuda"),
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    student.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                # Teacher predictions.
                mask = mask_sampler(len(input), seed=batch_idx).to(input.device)
                mask_ratio = len(mask) // len(input)
                input_repeat = input.repeat_interleave(mask_ratio, 0)
                masked_preds = teacher(input_repeat, mask)

                # Student reconstructions.
                reconstruction = student(input, mask)
                if loss_function == "mse":
                    loss = torch.nn.functional.mse_loss(reconstruction, masked_preds)
                elif loss_function == "cosine":
                    loss = 1 - torch.nn.functional.cosine_similarity(reconstruction, masked_preds, dim=-1).mean()
                elif loss_function == "l1":
                    loss = torch.nn.functional.l1_loss(reconstruction, masked_preds)
                elif loss_function == "masked_mse":
                    # Calculate loss for masked tokens only.
                    if reconstruction.shape[1] > mask.shape[1]:
                        num_prefix_tokens = reconstruction.shape[1] - mask.shape[1]
                        mask = torch.nn.functional.pad(mask, (num_prefix_tokens, 0), value=1)
                    loss = torch.mean((1 - mask).unsqueeze(-1) * (reconstruction - masked_preds) ** 2)
                elif loss_function == "unmasked_mse":
                    # Calculate loss for unmasked tokens only.
                    if reconstruction.shape[1] > mask.shape[1]:
                        num_prefix_tokens = reconstruction.shape[1] - mask.shape[1]
                        mask = torch.nn.functional.pad(mask, (num_prefix_tokens, 0), value=1)
                    loss = torch.mean(mask.unsqueeze(-1) * (reconstruction - masked_preds) ** 2)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), len(input))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                )

    metrics = OrderedDict([("loss", losses_m.avg)])

    return metrics


def get_baseline_loss(
    teacher,
    loader,
    mask_sampler,
    args,
    device=torch.device("cuda"),
    amp_autocast=suppress,
):
    # Calculate variance of targets as a baseline loss, as if we predicted the mean.
    n = 0
    mean = 0
    sum_squares = 0

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                # Teacher predictions.
                mask = mask_sampler(len(input), seed=batch_idx).to(input.device)
                mask_ratio = len(mask) // len(input)
                input_repeat = input.repeat_interleave(mask_ratio, 0)
                masked_preds = teacher(input_repeat, mask)

            if args.distributed:
                # Update running estimates.
                n += masked_preds.numel() * args.world_size
                diff = masked_preds.flatten() - mean
                reduced_diff = utils.reduce_tensor(diff.sum(), args.world_size)
                mean += reduced_diff.item() * args.world_size / n
                diff2 = masked_preds.flatten() - mean
                reduced_diff_product = utils.reduce_tensor((diff * diff2).sum(), args.world_size)
                sum_squares += reduced_diff_product.item() * args.world_size

            else:
                # Update running estimates.
                n += masked_preds.numel()
                diff = masked_preds.flatten() - mean
                mean += diff.sum().item() / n
                diff2 = masked_preds.flatten() - mean
                sum_squares += (diff * diff2).sum().item()

            if device.type == "cuda":
                torch.cuda.synchronize()

    variance_estimate = sum_squares / n
    metrics = OrderedDict([("baseline_loss", variance_estimate)])

    return metrics


if __name__ == "__main__":
    main()
