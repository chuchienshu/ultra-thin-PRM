import base64
# from typing import Tuple, List, Union, Dict, Iterable

import numpy as np
# import torch.nn as nn

def rle_encode(mask: np.ndarray) -> dict:
    """Perform Run-Length Encoding (RLE) on a binary mask.
    """

    assert mask.dtype == bool and mask.ndim == 2, 'RLE encoding requires a binary mask (dtype=bool).'
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return dict(data=base64.b64encode(runs.astype(np.uint32).tobytes()).decode('utf-8'), shape=mask.shape)


def rle_decode(rle: dict) -> np.ndarray:
    """Decode a Run-Length Encoding (RLE).
    """

    runs = np.frombuffer(base64.b64decode(rle['data']), np.uint32)
    shape = rle['shape']
    starts, lengths = [np.asarray(x, dtype=int) for x in (runs[0:][::2], runs[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
