"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""

from chemprop.train import chemprop_predict
from chemprop.nn_utils import *

if __name__ == '__main__':
    chemprop_predict()
