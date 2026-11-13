from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from algorithms.filters import binarize, gaussian_blur, sobel
from algorithms.morphology import closing, difference, opening

L_CODES = ("0001101", "0011001", "0010011", "0111101", "0100011",
           "0110001", "0101111", "0111011", "0110111", "0001011")
G_CODES = tuple(code[::-1].translate(str.maketrans("01", "10")) for code in L_CODES)
R_CODES = tuple(code.translate(str.maketrans("01", "10")) for code in L_CODES)
PARITIES = ("LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
            "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL")


def _digits(code, length):
    if not isinstance(code, str) or len(code) != length or any(c not in "0123456789" for c in code):
        raise ValueError(f"Ожидается строка из {length} цифр 0–9")


def checksum(digits12):
    _digits(digits12, 12)
    return str((-sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits12))) % 10)


def normalize_code(code):
    if isinstance(code, str) and len(code) == 12:
        return code + checksum(code)
    _digits(code, 13)
    if checksum(code[:12]) != code[12]:
        raise ValueError("Неверная контрольная цифра EAN-13")
    return code


def encode_bits(code):
    code = normalize_code(code)
    left = "".join((L_CODES if parity == "L" else G_CODES)[int(digit)]
                   for digit, parity in zip(code[1:7], PARITIES[int(code[0])]))
    return "101" + left + "01010" + "".join(R_CODES[int(d)] for d in code[7:]) + "101"


def decode_bits(bits):
    if len(bits) != 95 or bits[:3] != "101" or bits[45:50] != "01010" or bits[-3:] != "101":
        raise ValueError("Неверные защитные последовательности EAN-13")
    digits, parity = "", ""
    for i in range(3, 45, 7):
        word = bits[i:i + 7]
        if word in L_CODES:
            digits += str(L_CODES.index(word))
            parity += "L"
        elif word in G_CODES:
            digits += str(G_CODES.index(word))
            parity += "G"
        else:
            raise ValueError("Не распознана цифра левой половины")
    if parity not in PARITIES:
        raise ValueError("Неверная чётность левой половины")
    code = str(PARITIES.index(parity)) + digits
    for i in range(50, 92, 7):
        word = bits[i:i + 7]
        if word not in R_CODES:
            raise ValueError("Не распознана цифра правой половины")
        code += str(R_CODES.index(word))
    return normalize_code(code)


def generate(code, module_width=4, bar_height=160, margin=40):
    if any(not isinstance(v, int) for v in (module_width, bar_height, margin)):
        raise ValueError("Размеры штрихкода должны быть целыми")
    if module_width < 1 or bar_height < 1 or margin < 0:
        raise ValueError("Некорректные размеры штрихкода")
    bits = encode_bits(code)
    image = Image.new("RGB", (95 * module_width + 2 * margin, bar_height + 2 * margin), "white")
    painter = ImageDraw.Draw(image)
    for i, bit in enumerate(bits):
        if bit == "1":
            x = margin + i * module_width
            painter.rectangle((x, margin, x + module_width - 1, margin + bar_height - 1), fill="black")
    return image


def pipeline_mask(image):
    edges = binarize(gaussian_blur(sobel(image.convert("L")), 5, 1.5), 50)
    return difference(opening(edges, 5), closing(edges, 5))


@dataclass
class BarcodeResult:
    code: str
    box: tuple
    scan_y: int
    image: Image.Image


def _decode_row(dark):
    changes = np.flatnonzero(dark[1:] != dark[:-1]) + 1
    edges = np.r_[0, changes, len(dark)]
    for i in range(len(edges) - 59):
        if not dark[edges[i]]:
            continue
        start, stop = int(edges[i]), int(edges[i + 59])
        runs = np.diff(edges[i:i + 60])
        module = (stop - start) / 95
        guards = runs[np.r_[0:3, 27:32, 56:59]]
        if module < 0.8 or np.max(np.abs(guards / module - 1)) > 0.7:
            continue
        centers = start + (np.arange(95) + 0.5) * module
        bits = "".join("1" if dark[int(v)] else "0" for v in centers)
        for candidate in (bits, bits[::-1]):
            try:
                return decode_bits(candidate), start, stop
            except ValueError:
                pass
    return None


def analyze(image):
    gray = np.asarray(image.convert("L"))
    mask = np.asarray(pipeline_mask(image)) > 0
    ys, _ = np.where(mask)
    preferred = [int(np.median(ys))] if len(ys) else []
    rows = preferred + list(np.linspace(0, image.height - 1, min(41, image.height), dtype=int))
    rows = sorted(set(rows), key=lambda y: abs(y - image.height / 2))
    for y in rows:
        row = gray[y]
        low, high = int(row.min()), int(row.max())
        if high - low < 40:
            continue
        found = _decode_row(row < (low + high) / 2)
        if found is None:
            continue
        code, left, right = found
        strip = gray[:, left:right]
        dark = strip < (low + high) / 2
        transitions = np.count_nonzero(dark[:, 1:] != dark[:, :-1], axis=1)
        top, bottom = y, y
        while top > 0 and transitions[top - 1] >= 40:
            top -= 1
        while bottom + 1 < image.height and transitions[bottom + 1] >= 40:
            bottom += 1
        annotated = image.convert("RGB").copy()
        painter = ImageDraw.Draw(annotated)
        painter.rectangle((left, top, right - 1, bottom), outline=(30, 180, 70), width=2)
        painter.line((left, y, right - 1, y), fill=(230, 60, 60), width=1)
        return BarcodeResult(code, (left, top, right, bottom + 1), int(y), annotated)
    raise ValueError("EAN-13 не найден. Нужны чёткие горизонтальные штрихи без перспективного искажения")
