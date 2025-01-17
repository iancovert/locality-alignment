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
import torch.nn as nn
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
from timm.layers import AttentionPoolLatent
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler as GradScaler
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy

from locality_alignment import (
    load_checkpoint_auto,
    CheckpointSaver,
    TransformerPooling,
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
group.add_argument("--model", default="vit_base_patch16_224.augreg_in21k_ft_in1k", type=str, metavar="MODEL", help='Name of model to train (default: "vit_base_patch16_224.augreg_in21k_ft_in1k")')
group.add_argument("--pretrained", type=bool, default=True, help="Start with pretrained version of specified network (if avail)")
group.add_argument("--output-head", default="linear", type=str, help='type of output head (default: "linear", "attention" or "transformer")')
group.add_argument("--freeze-backbone", default=False, type=bool, help="whether to freeze the backbone and only learn the output head")
group.add_argument("--initial-checkpoint", default="", type=str, metavar="PATH", help="Load this checkpoint into model after initialization (default: none)")
group.add_argument("--resume", default="", type=str, metavar="PATH", help="Resume full model and optimizer state from checkpoint (default: none)")
group.add_argument("--no-resume-opt", action="store_true", default=False, help="prevent resume of optimizer state when resuming model")
group.add_argument("--num-classes", type=int, default=None, metavar="N", help="number of label classes (Model default if None)")
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
group.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")
group.add_argument("--train-interpolation", type=str, default="random", help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")
group.add_argument("--drop-path", type=float, default=None, metavar="PCT", help="Drop path rate (default: None)")
group.add_argument("--drop-block", type=float, default=None, metavar="PCT", help="Drop block rate (default: None)")

# Loss function parameters.
group = parser.add_argument_group("Loss function parameters")
group.add_argument("--jsd-loss", action="store_true", default=False, help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.")
group.add_argument("--bce-loss", action="store_true", default=False, help="Enable BCE loss w/ Mixup/CutMix use.")
group.add_argument("--bce-sum", action="store_true", default=False, help="Sum over classes when using BCE loss.")
group.add_argument("--bce-target-thresh", type=float, default=None, help="Threshold for binarizing softened BCE targets (default: None, disabled).")
group.add_argument("--bce-pos-weight", type=float, default=None, help="Positive weighting for BCE loss.")

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

    # Create model.
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )

    # Initialize from checkpoint.
    if args.initial_checkpoint:
        head, model.head = model.head, nn.Identity()
        non_matching_keys = load_checkpoint_auto(model, args.initial_checkpoint, strict=False)
        model.head = head
        if utils.is_primary(args):
            _logger.info(f"Loaded {args.initial_checkpoint}, non-matching keys: {non_matching_keys}")

    # Create output head.
    if args.output_head == "linear":
        # Switch to global pooling.
        model.global_pool = "avg"
        model.attn_pool = None

        # Zero-init linear output head.
        if args.freeze_backbone:
            nn.init.zeros_(model.head.weight)
            nn.init.zeros_(model.head.bias)
    elif args.output_head == "attention":
        # Switch to attention pooling.
        model.global_pool = None
        num_heads = model.blocks[0].attn.num_heads
        mlp_ratio = model.blocks[0].mlp.fc1.weight.shape[0] / model.blocks[0].mlp.fc1.weight.shape[1]
        model.attn_pool = AttentionPoolLatent(
            model.embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=type(model.norm),
        )

        # No fc norm.
        model.fc_norm = nn.Identity()
    elif args.output_head == "transformer":
        # Switch to transformer pooling.
        model.global_pool = None
        num_heads = model.blocks[0].attn.num_heads
        num_tokens = model.patch_embed.num_patches + model.num_prefix_tokens
        model.attn_pool = TransformerPooling(
            embed_dim=model.embed_dim,
            num_heads=num_heads,
            num_tokens=num_tokens,
            num_layers=1,
        )

        # No fc norm.
        model.fc_norm = nn.Identity()
    else:
        raise ValueError(f"Unrecognized output head type: {args.output_head}")

    # Freeze backbone.
    if args.freeze_backbone:
        # Turn off grads for all params except head.
        for param in set(model.parameters()) - set(model.head.parameters()):
            param.requires_grad = False

        # Turn on attention pooling gradients.
        if model.attn_pool:
            for param in model.attn_pool.parameters():
                param.requires_grad = True

    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # Setup augmentation batch splits for split bn.
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # Move model to GPU, enable channels last layout if set.
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not args.torchcompile
        model = torch.jit.script(model)

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

    optimizer = create_optimizer_v2(
        filter(lambda p: p.requires_grad, model.parameters()),
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
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # Set up EMA of model weights, SWA could be used here too.
    model_ema = None
    if args.model_ema:
        # Create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper.
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint_auto(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # Set up distributed training.
    if args.distributed:
        model = DDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA and teacher don't need to be wrapped by DDP.

    if args.torchcompile:
        # Torch compile should be done after DDP.
        assert has_compile, "A version of torch with torch.compile() is required for --compile."
        model = torch.compile(model, backend=args.torchcompile)

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
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
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

    # Set up loss function.
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

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
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(args.output if args.output else "./output/train", exp_name)
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
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

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                train_loss_fn,
                optimizer,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
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
    model,
    loader,
    loss_fn,
    optimizer,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

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
                output = model(input)
                loss = loss_fn(output, target)

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
                    parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head="agc" in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
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
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

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
    model,
    loader,
    loss_fn,
    args,
    device=torch.device("cuda"),
    amp_autocast=suppress,
    log_suffix="",
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

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
                output = model(input)
                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), len(input))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                )

    metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])

    return metrics


if __name__ == "__main__":
    main()
