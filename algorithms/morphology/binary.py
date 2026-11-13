import numpy as np
from PIL import Image

from algorithms.filters.spatial import _kernel_size


def _morph(image, size, operation):
    size = _kernel_size(size)
    mask = np.asarray(image.convert("L")) > 127
    padded = np.pad(mask, size // 2, mode="edge")
    result = np.zeros_like(mask) if operation == "dilate" else np.ones_like(mask)
    for y in range(size):
        for x in range(size):
            part = padded[y:y + mask.shape[0], x:x + mask.shape[1]]
            if operation == "dilate":
                result |= part
            else:
                result &= part
    return Image.fromarray(result.astype(np.uint8) * 255)


def dilate(image, kernel_size=3):
    return _morph(image, kernel_size, "dilate")


def erode(image, kernel_size=3):
    return _morph(image, kernel_size, "erode")


def opening(image, kernel_size=3):
    return dilate(erode(image, kernel_size), kernel_size)


def closing(image, kernel_size=3):
    return erode(dilate(image, kernel_size), kernel_size)


def difference(first, second):
    if first.size != second.size:
        raise ValueError("Размеры масок должны совпадать")
    a = np.asarray(first.convert("L")) > 127
    b = np.asarray(second.convert("L")) > 127
    return Image.fromarray(np.logical_xor(a, b).astype(np.uint8) * 255)
