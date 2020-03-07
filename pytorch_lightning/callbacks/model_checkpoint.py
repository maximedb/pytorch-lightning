"""
Model Checkpointing
===================

Automatically save model checkpoints during training.
"""

import os
import shutil
import logging as log
import warnings
import re

import numpy as np

from pytorch_lightning.callbacks.base import Callback


class ModelCheckpoint(Callback):
    r"""
    Save the model after every epoch.

    Args:
        filepath: path to save the model file.
            Can contain named formatting options to be auto-filled.

            Example::

                # no path
                ModelCheckpoint()
                #  saves like /my/path/epoch_0.ckpt

                # save any arbitrary metrics like and val_loss, etc in name
                ModelCheckpoint(filepath='/my/path/{epoch}-{val_loss:.2f}-{other_metric:.2f}')
                # saves file like: /my/path/epoch=2-val_loss=0.2_other_metric=0.3.ckpt


        monitor (str): quantity to monitor.
        verbose (bool): verbosity mode, False or True.
        save_top_k (int): if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if ``save_top_k == 0``, no models are saved.
            if ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if ``save_top_k >= 2`` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode (str): one of {auto, min, max}.
            If ``save_top_k != 0``, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only (bool): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period (int): Interval (number of epochs) between checkpoints.

    Example::

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        # saves checkpoints to my_path whenever 'val_loss' has a new min
        checkpoint_callback = ModelCheckpoint(filepath='my_path')
        Trainer(checkpoint_callback=checkpoint_callback)

        # save epoch and val_loss in name
        ModelCheckpoint(filepath='/my/path/here/sample-mnist_{epoch:02d}-{val_loss:.2f}')
        # saves file like: /my/path/here/sample-mnist_epoch=02_val_loss=0.32.ckpt
    """

    def __init__(self, filepath, monitor: str = 'val_loss', verbose: bool = False,
                 save_top_k: int = 1, save_weights_only: bool = False,
                 mode: str = 'auto', period: int = 1, prefix: str = ''):
        super().__init__()
        # if save_top_k and os.path.isdir(filepath) and len(os.listdir(filepath)) > 0:
        #     warnings.warn(
        #         f"Checkpoint directory {filepath} exists and is not empty with save_top_k != 0."
        #         "All files in this directory will be deleted when a checkpoint is saved!"
        #     )
        if os.path.isdir(filepath):
            self.dirpath, self.filename = filepath, '{epoch}'
        else:
            self.dirpath, self.filename = os.path.split(filepath)

        # os.makedirs(self.dirpath, exist_ok=True)
        # self.save_top_k = save_top_k
        # self.save_weights_only = save_weights_only
        # self.period = period
        # self.epochs_since_last_check = 0
        self.prefix = prefix
        # self.best_k_models = {}
        # # {filename: monitor}
        # self.kth_best_model = ''
        # self.best = 0
        self.save_function = None

        # mode_dict = {
        #     'min': (np.less, np.Inf, 'min'),
        #     'max': (np.greater, -np.Inf, 'max'),
        #     'auto': (np.greater, -np.Inf, 'max') if 'acc' in self.monitor or self.monitor.startswith('fmeasure')
        #     else (np.less, np.Inf, 'min'),
        # }

        # if mode not in mode_dict:
        #     warnings.warn(
        #         f'ModelCheckpoint mode {mode} is unknown, '
        #         'fallback to auto mode.', RuntimeWarning)
        #     mode = 'auto'

        # self.monitor_op, self.kth_value, self.mode = mode_dict[mode]

    def format_checkpoint_name(self, epoch, metrics, ver=None):
        """Generate a filename according define template.

        Examples
        --------
        >>> tmpdir = os.path.dirname(__file__)
        >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}'))
        >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
        'epoch=0.ckpt'
        >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch:03d}'))
        >>> os.path.basename(ckpt.format_checkpoint_name(5, {}))
        'epoch=005.ckpt'
        >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}-{val_loss:.2f}'))
        >>> os.path.basename(ckpt.format_checkpoint_name(2, dict(val_loss=0.123456)))
        'epoch=2-val_loss=0.12.ckpt'
        >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{missing:d}'))
        >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
        'missing=0.ckpt'
        """
        # check if user passed in keys to the string
        groups = re.findall(r'(\{.*?)[:\}]', self.filename)

        if len(groups) == 0:
            # default name
            filename = f'{self.prefix}_ckpt_epoch_{epoch}'
        else:
            metrics['epoch'] = epoch
            filename = self.filename
            for tmp in groups:
                name = tmp[1:]
                filename = filename.replace(tmp, name + '={' + name)
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)
        str_ver = f'_v{ver}' if ver is not None else ''
        filepath = os.path.join(self.dirpath, self.prefix + filename + str_ver + '.ckpt')
        return filepath

    def on_validation_end(self, trainer, pl_module):
        if trainer.use_tpu:
            if pl_module.tpu_global_core_rank == 0:
                metrics = trainer.callback_metrics
                epoch = trainer.current_epoch
                filepath = self.format_checkpoint_name(epoch, metrics)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                self.save_function(filepath)
