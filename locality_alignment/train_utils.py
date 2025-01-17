"""Checkpoint saving utilities.

Modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/checkpoint_saver.py
and https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_helpers.py
"""

import os
import timm
import torch
import logging
import operator
from typing import Union, Optional, Any, Callable
from timm.utils.model import unwrap_model, get_state_dict
from .model_utils import MaskEmbedStudent


try:
    import safetensors.torch

    _has_safetensors = True
except ImportError:
    _has_safetensors = False


_logger = logging.getLogger(__name__)


class CheckpointSaver(timm.utils.CheckpointSaver):
    """Checkpoint saver modified to retain results at a regular interval."""

    def __init__(
        self,
        model,
        optimizer,
        args=None,
        model_ema=None,
        amp_scaler=None,
        checkpoint_prefix="checkpoint",
        recovery_prefix="recovery",
        checkpoint_dir="",
        recovery_dir="",
        decreasing=False,
        max_history=10,
        retain_interval=None,
        unwrap_fn=unwrap_model,
    ):
        # Objects whose state_dicts to save.
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # Track current checkpoints, (filename, metric, epoch) tuples sorted by metric.
        self.checkpoint_files = []
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ""
        self.last_recovery_file = ""

        # Configure for saving.
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = ".pth.tar"
        self.max_history = max_history
        self.retain_interval = retain_interval
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

        # Set comparison operator for metric.
        self.decreasing = decreasing
        self.cmp = operator.lt if decreasing else operator.gt

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, "tmp" + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, "last" + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # Required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or metric is None or self.cmp(metric, worst_file[1]):
            # Clear worst checkpoint if necessary.
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            # Save current checkpoint and sort checkpoint files.
            filename = "-".join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric, epoch))
            self.checkpoint_files = sorted(self.checkpoint_files, key=lambda x: x[1], reverse=not self.decreasing)

            # Log current checkpoints.
            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += " {}\n".format(c)
            _logger.info(checkpoints_str)

            # Update best checkpoint.
            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, "model_best" + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            "epoch": epoch,
            "arch": type(self.model).__name__.lower(),
            "state_dict": get_state_dict(self.model, self.unwrap_fn),
            "optimizer": self.optimizer.state_dict(),
            "version": 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state["arch"] = self.args.model
            save_state["args"] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state["state_dict_ema"] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state["metric"] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            # Skip removal if epoch matches retain frequency.
            if (self.retain_interval is not None) and ((d[2] + 1) % self.retain_interval == 0):
                _logger.debug("Retaining checkpoint: {}".format(d))
            else:
                # Delete checkpoint.
                try:
                    _logger.debug("Cleaning checkpoint: {}".format(d))
                    os.remove(d[0])
                except Exception as e:
                    _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]


def auto_filter_fn(state_dict: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    """Perform automatic filtering of state dict keys based on model."""
    if isinstance(model, MaskEmbedStudent):
        # Loading into MaskEmbedStudent model, do nothing.
        pass

    elif not all([key.split(".")[0] in ["encoder", "decoder"] for key in state_dict.keys()]):
        # Checkpoint isn't MaskEmbedStudent model, do nothing.
        pass

    else:
        # Loading into vision backbone, extract encoder keys only.
        logging.info("Filtering encoder keys from checkpoint state_dict")
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

        # Remove layer scale if necessary.
        ls_keys = [k for k in state_dict.keys() if (".ls1" in k or ".ls2" in k)]
        target_state_dict = model.state_dict()
        removed_ls = False
        for k in ls_keys:
            if k not in target_state_dict:
                # Determine target params in either attn or mlp module.
                if ".ls1" in k:
                    weight_key = k.replace(".ls1.gamma", ".attn.proj.weight")
                    bias_key = k.replace(".ls1.gamma", ".attn.proj.bias")
                else:
                    weight_key = k.replace(".ls2.gamma", ".mlp.fc2.weight")
                    bias_key = k.replace(".ls2.gamma", ".mlp.fc2.bias")

                # Absorb layer scale into previous layer.
                if weight_key in state_dict:
                    state_dict[weight_key] *= state_dict[k].unsqueeze(-1)
                if bias_key in state_dict:
                    state_dict[bias_key] *= state_dict[k]

                # Cleanup.
                del state_dict[k]
                removed_ls = True

        if removed_ls:
            logging.info("Removed layer scale from checkpoint state_dict")

        # Check if model has pooling and override.
        if hasattr(model, "attn_pool") and model.attn_pool is not None:
            logging.info("Setting attention pooling to None")
            model.attn_pool = None
        if model.global_pool is not None:
            logging.info("Setting global pooling to None")
            model.global_pool = None

    return state_dict


def load_checkpoint_auto(
    model: torch.nn.Module,
    checkpoint_path: str,
    use_ema: bool = True,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
):
    return load_checkpoint(
        model,
        checkpoint_path,
        use_ema,
        device,
        strict,
        auto_filter_fn,
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    use_ema: bool = True,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
    filter_fn: Optional[Callable] = None,
):
    # No support for numpy checkpoints.
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        raise NotImplementedError("Model cannot load numpy checkpoint")

    # Load state dict.
    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if filter_fn:
        state_dict = filter_fn(state_dict, model)

    # Load state dict into model.
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def load_state_dict(
    checkpoint_path: str,
    use_ema: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly.
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get state dict key.
        state_dict_key = ""
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get("state_dict_ema", None) is not None:
                state_dict_key = "state_dict_ema"
            elif use_ema and checkpoint.get("model_ema", None) is not None:
                state_dict_key = "model_ema"
            elif "state_dict" in checkpoint:
                state_dict_key = "state_dict"
            elif "model" in checkpoint:
                state_dict_key = "model"

        # Clean checkpoint from unwanted key prefixes.
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    # Remove `module.` prefix from parallel training and `_orig_mod.` prefix from dynamo.
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        k = k[10:] if k.startswith("_orig_mod.") else k
        k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[k] = v
    return cleaned_state_dict
