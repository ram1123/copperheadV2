#!/usr/bin/env python3
"""Importable Ray driver for four-fold VBF-DNN training.

This module must remain separate from ``train_dnn``.  Executing ``train_dnn``
itself as ``__main__`` makes Ray cloudpickle its task functions and their torch
globals, including the non-serializable CudnnModule object.  Importing it here
keeps those task functions addressable by their real module name.
"""

from MVA_training.VBF_run3 import train_dnn


if __name__ == "__main__":
    train_dnn.main()
