"""Minimal TensorBoard logger.

`train.py` and `train_regression.py` both do `from logger import Logger` and expect a
`Logger` class with the same interface as the well-known TensorBoard logger used in
`yunjey/pytorch-tutorial` (a `Logger(log_dir)` constructor plus `scalar_summary`,
`histo_summary`, and `image_summary` methods). That `logger.py` was never checked into
the original project, so both training scripts would fail immediately on import.

This is a minimal, drop-in reimplementation of that same interface, built on top of
`torch.utils.tensorboard.SummaryWriter` (part of PyTorch; needs the separate
`tensorboard` package installed - see requirements.txt) instead of the original's
raw TensorFlow summary protobufs. Usage is unchanged from what both scripts already do:

    logger = Logger('./logs')
    logger.scalar_summary(tag, value, step)
    logger.histo_summary(tag, values, step)          # values: a numpy array
    logger.image_summary(tag, images, step)          # images: (N, C, H, W) or (N, H, W, C)

Run `tensorboard --logdir <log_dir>` to view the results.
"""

from __future__ import annotations

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer that logs to `log_dir` (created if missing)."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a single scalar (e.g. loss, accuracy) at a given training step."""
        self.writer.add_scalar(tag, value, global_step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of a numpy array's values at a given training step.

        `bins` matches the original interface (max number of histogram bins);
        it is forwarded to SummaryWriter's `max_bins` argument.
        """
        values = np.asarray(values)
        self.writer.add_histogram(tag, values, global_step=step, max_bins=bins)

    def image_summary(self, tag, images, step):
        """Log a batch of images at a given training step.

        `images` is a numpy array shaped either (N, C, H, W) or (N, H, W, C), with
        C in {1, 3}. Values may be uint8 in [0, 255] or float in any range - floats
        are rescaled into [0, 1] here so SummaryWriter doesn't silently overflow
        when it casts to uint8 internally (this matters for callers that pass
        normalized model-input tensors rather than display-ready images).
        """
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(
                f"image_summary expects a 4D array (N,C,H,W) or (N,H,W,C); got shape {images.shape}"
            )
        dataformats = "NCHW" if images.shape[1] in (1, 3) else "NHWC"

        if images.dtype != np.uint8:
            images = images.astype(np.float32)
            lo, hi = float(images.min()), float(images.max())
            images = (images - lo) / (hi - lo) if hi > lo else np.zeros_like(images)

        self.writer.add_images(tag, images, global_step=step, dataformats=dataformats)

    def close(self):
        """Flush and close the underlying SummaryWriter."""
        self.writer.close()
