import numpy as np
from PIL import Image


def _kernel_size(size):
    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError("Размер ядра должен быть положительным нечётным числом")
    return size


def correlate(array, kernel):
    array = np.asarray(array, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    padding = [(kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2)]
    if array.ndim == 3:
        padding.append((0, 0))
    padded = np.pad(array, padding, mode="edge")
    result = np.zeros_like(array, dtype=float)
    for y in range(kh):
        for x in range(kw):
            if kernel[y, x]:
                result += padded[y:y + array.shape[0], x:x + array.shape[1]] * kernel[y, x]
    return result


def _channels(image):
    mode = image.mode if image.mode in ("L", "RGB", "RGBA") else "RGB"
    array = np.asarray(image.convert(mode))
    return (array[..., :3] if mode == "RGBA" else array), mode, array


def _image(values, mode, original):
    result = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    if mode == "RGBA":
        result = np.dstack((result, original[..., 3]))
    return Image.fromarray(result)


def box_blur(image, kernel_size=3):
    size = _kernel_size(kernel_size)
    array, mode, original = _channels(image)
    return _image(correlate(array, np.ones((size, size)) / size**2), mode, original)


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    size = _kernel_size(kernel_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Сигма должна быть положительным конечным числом")
    offsets = np.arange(size) - size // 2
    kernel = np.exp(-(offsets[:, None]**2 + offsets[None, :]**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    array, mode, original = _channels(image)
    return _image(correlate(array, kernel), mode, original)


def _gradient(image, kernel):
    array, mode, original = _channels(image)
    gx, gy = correlate(array, kernel), correlate(array, np.asarray(kernel).T)
    return _image(np.hypot(gx, gy), mode, original)


def sobel(image):
    return _gradient(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def prewitt(image):
    return _gradient(image, [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])


def scharr(image):
    return _gradient(image, [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])


def laplacian(image):
    array, mode, original = _channels(image)
    return _image(np.abs(correlate(array, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])), mode, original)


def bin_filter(image):
    array, mode, original = _channels(image)
    return _image(np.abs(correlate(array, [[1, -1], [-1, 1]])), mode, original)


def binarize(image, threshold=128):
    if not 0 <= threshold <= 255:
        raise ValueError("Порог должен лежать в диапазоне 0...255")
    gray = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2)
    return Image.fromarray(np.where(gray > threshold, 255, 0).astype(np.uint8))


def sobel_filter(image, width, height):
    if width < 1 or height < 1:
        raise ValueError("Размеры изображения должны быть положительными")
    array = np.asarray(list(image), dtype=float)
    if array.size != width * height:
        raise ValueError("Число пикселей не соответствует размерам изображения")
    array = array.reshape(height, width)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    magnitude = np.hypot(correlate(array, kernel), correlate(array, kernel.T))
    return np.clip(magnitude, 0, 255).astype(np.uint8).ravel().tolist()
