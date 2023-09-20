import random

import torch

import librosa as lb
import numpy as np
from loguru import logger
import albumentations


def set_seed(seed=1771):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logger.info(f'Set seed: {seed}')


def crop_or_pad(y, length, start=None):
    """
    Crop or padding for train audio
    :param y:
    :param length:
    :param start:
    :return:
    """
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        n_repeats = length // len(y)
        epsilon = length % len(y)
        y = np.concatenate([y] * n_repeats + [y[:epsilon]])

    elif len(y) > length:
        start = start or np.random.randint(len(y) - length)
        y = y[start:start + length]

    return y


def compute_melspec(y, sr, n_mels, fmin, fmax, n_fft=2048, hop_length=512):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    if fmax is None:
        fmax = sr // 2


    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, n_fft=n_fft, hop_length=hop_length
    )

    melspec = lb.power_to_db(melspec.astype(np.float32), ref=np.max)
    return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def normalize(image):
    image = image.astype(np.uint8)
    image = np.stack([image, image, image], axis=-1)
    transform = albumentations.Compose([
        albumentations.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
    ])
    return transform(image=image)['image'].T