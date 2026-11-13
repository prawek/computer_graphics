from dataclasses import dataclass, field
from math import cos, pi, sin
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from algorithms.barcode import analyze, generate, normalize_code, pipeline_mask
from algorithms.clipping import clip_segment
from algorithms.filling import flood_fill, scanline_fill, span_fill
from algorithms.filters import (binarize, bin_filter, box_blur, gaussian_blur,
                                laplacian, prewitt, scharr, sobel)
from algorithms.fractals import minkowski_island
from algorithms.geometry import Point3
from algorithms.morphology import closing, difference, dilate, erode, opening
from algorithms.rasterization import draw, draw_circle, draw_line
from algorithms.rendering import draw_buffer, draw_light, fill3D, get_plane, update_buffer
from algorithms.transformations import rotate2D, rotate3D, scale2D, shift2D, shift3D

SIZE = (640, 480)
BACKGROUND = (20, 27, 42)
BLUE, GREEN, ORANGE = (80, 165, 255), (60, 215, 160), (255, 170, 70)


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    default: object
    minimum: float = 0
    maximum: float = 0
    step: float = 1


@dataclass(frozen=True)
class Demo:
    key: str
    group: str
    title: str
    description: str
    folder: str
    parameters: tuple = ()
    accepts_image: bool = False


@dataclass
class Result:
    original: Image.Image
    processed: Image.Image
    info: str = ""
    elapsed_ms: float = field(default=0.0)


KERNEL = Parameter("kernel", "Ядро (нечётное)", 5, 1, 15, 2)
THRESHOLD = Parameter("threshold", "Порог", 128, 0, 255)
ANGLE = Parameter("angle", "Угол, градусы", 30, -180, 180, 5)
LEVEL = Parameter("level", "Итерации", 2, 0, 4)
CODE = Parameter("code", "EAN: 12 или 13 цифр", "590123412345")

DEMOS = [
    Demo("line", "Растеризация", "Отрезок Брезенхема", "Выбор следующего пикселя по накопленной целочисленной ошибке. Показаны все октанты.", "rasterization",
         (Parameter("rays", "Количество лучей", 32, 4, 96, 4),)),
    Demo("circle", "Растеризация", "Окружность Брезенхема", "Расчёт одного октанта методом средней точки и отражение по восьми направлениям.", "rasterization",
         (Parameter("radius", "Радиус", 150, 0, 220, 10),)),
    Demo("clipping", "Отсечение", "Кирус — Бек", "Серые линии — исходные отрезки, зелёные — видимая часть внутри выпуклого контура.", "clipping"),
    Demo("scanline", "Заливки", "Сканирующая строка / AEL", "Таблица рёбер и активный список пересечений. Поддерживает невыпуклые многоугольники.", "filling"),
    Demo("flood", "Заливки", "Затравка / BFS", "Очередь обходит четырёхсвязную область. Щёлкните на исходнике, чтобы изменить затравку.", "filling"),
    Demo("span", "Заливки", "Интервальная заливка", "Вместо отдельных соседей в стек добавляются затравки горизонтальных полос. Щёлкните на исходнике.", "filling"),
    Demo("fill_compare", "Заливки", "Сравнение трёх заливок", "Один контур для BFS, интервалов и AEL. Число пикселей может отличаться из-за правила включения границы.", "filling", (LEVEL,)),
    Demo("transform2d", "Преобразования", "Аффинные 2D", "Поворот, масштаб и перенос через матрицы 3×3. В правом окне серым показан исходный контур.", "transformations",
         (ANGLE, Parameter("scale", "Масштаб", 1.1, 0.2, 1.8, 0.1), Parameter("dx", "Сдвиг X", 50, -150, 150, 10))),
    Demo("transform3d", "Преобразования", "Поворот 3D", "Вращение куба вокруг осей Y и X матрицами 4×4; ортографическая проекция.", "transformations", (ANGLE,)),
    Demo("zbuffer", "Рендеринг", "Z-буфер", "Две пересекающиеся грани: для каждого пикселя остаётся ближайшая глубина z.", "rendering"),
    Demo("lambert", "Рендеринг", "Свет Ламберта", "Яркость пропорциональна косинусу угла между нормалью и направлением к свету.", "rendering",
         (Parameter("light_x", "Источник света X", 260, 0, 640, 20),)),
    Demo("phong", "Рендеринг", "Блик Фонга", "К диффузному свету добавляется зеркальная составляющая по направлению отражённого луча.", "rendering",
         (Parameter("light_x", "Источник света X", 260, 0, 640, 20),)),
    Demo("box", "Фильтры", "Усредняющий", "Среднее в квадратном окне. Граничные пиксели повторяются, тёмной рамки не возникает.", "filters", (KERNEL,), True),
    Demo("gaussian", "Фильтры", "Гаусс", "Взвешенное среднее: ближние пиксели имеют больший вес. Сумма весов равна единице.", "filters",
         (KERNEL, Parameter("sigma", "Сигма", 1.5, 0.2, 5, 0.1)), True),
    Demo("sobel", "Фильтры", "Собель", "Два ядра 3×3 находят горизонтальный и вертикальный градиенты. Результат — их модуль.", "filters", accepts_image=True),
    Demo("prewitt", "Фильтры", "Прьюитт", "Градиент 3×3 с равномерными весами вдоль направления сглаживания.", "filters", accepts_image=True),
    Demo("laplacian", "Фильтры", "Лаплас", "Модуль второй производной с четырьмя соседями выделяет перепады яркости.", "filters", accepts_image=True),
    Demo("bin2x2", "Фильтры", "Диагональный 2×2", "Модуль отклика ядра [[1, −1], [−1, 1]] из C++ репозитория.", "filters", accepts_image=True),
    Demo("scharr", "Фильтры", "Шарр", "Градиент с весами 3 и 10; более сильный отклик, чем у Собеля.", "filters", accepts_image=True),
    Demo("binary", "Фильтры", "Бинаризация", "Средняя яркость RGB сравнивается с порогом; результат содержит только 0 и 255.", "filters", (THRESHOLD,), True),
    Demo("dilate", "Морфология", "Дилатация", "Белая область расширяется: достаточно одного белого пикселя в окне.", "morphology", (KERNEL, THRESHOLD), True),
    Demo("erode", "Морфология", "Эрозия", "Белый пиксель остаётся только тогда, когда всё окно белое.", "morphology", (KERNEL, THRESHOLD), True),
    Demo("opening", "Морфология", "Открытие", "Эрозия, затем дилатация: убирает небольшие белые детали.", "morphology", (KERNEL, THRESHOLD), True),
    Demo("closing", "Морфология", "Закрытие", "Дилатация, затем эрозия: заполняет небольшие чёрные разрывы.", "morphology", (KERNEL, THRESHOLD), True),
    Demo("diff", "Морфология", "Разность масок / XOR", "Пиксели, в которых открытие и закрытие бинарной маски различаются.", "morphology", (KERNEL, THRESHOLD), True),
    Demo("ean_generate", "Штрихкод EAN-13", "Генерация", "Контрольная цифра, L/G/R-кодирование, защитные полосы; 95 модулей данных.", "barcode", (CODE,)),
    Demo("ean_decode", "Штрихкод EAN-13", "Поиск и чтение", "Маска морфологии и сканирование строк. Поддерживаются чёткие горизонтальные штрихкоды.", "barcode", (CODE,), True),
    Demo("ean_mask", "Штрихкод EAN-13", "Маска локализации", "Собель → Гаусс → бинаризация → XOR открытия и закрытия.", "barcode", (CODE,), True),
    Demo("minkowski", "Фракталы", "Остров Минковского", "Каждое ребро заменяется восемью сегментами. Геометрическая площадь остаётся постоянной.", "fractals", (LEVEL,)),
]
DEMO_BY_KEY = {demo.key: demo for demo in DEMOS}


def outline(image, vertices, color=BLUE):
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        draw(image, draw_line(*a, *b), color)


def sample_image():
    yy, xx = np.mgrid[:SIZE[1], :SIZE[0]]
    rng = np.random.default_rng(17)
    base = np.stack((50 + xx * 0.22, 65 + yy * 0.18, 140 + xx * 0.08), axis=2)
    image = Image.fromarray(np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8))
    painter = ImageDraw.Draw(image)
    painter.rectangle((60, 70, 265, 270), fill=(235, 190, 75))
    painter.ellipse((320, 110, 540, 330), fill=(65, 210, 170))
    painter.polygon([(150, 390), (350, 225), (500, 420)], fill=(220, 75, 110))
    for x in range(35, 610, 18):
        painter.line((x, 440, x, 465), fill="white", width=3)
    return image


def mask_image():
    image = Image.new("L", SIZE, 0)
    painter = ImageDraw.Draw(image)
    painter.rectangle((80, 90, 300, 320), fill=255)
    painter.ellipse((335, 110, 555, 350), fill=255)
    painter.rectangle((173, 90, 178, 195), fill=0)
    painter.ellipse((400, 175, 405, 180), fill=0)
    rng = np.random.default_rng(42)
    for x, y in rng.integers([25, 25], [610, 450], (65, 2)):
        painter.rectangle((int(x), int(y), int(x) + 1, int(y) + 1), fill=255)
    return image


def _fill_contour(vertices):
    image = Image.new("RGB", SIZE, BACKGROUND)
    outline(image, vertices, (230, 235, 245))
    return image


def _render_scene(key, light_x=260):
    image = Image.new("RGB", SIZE, BACKGROUND)
    faces = [[Point3(125, 90, 90), Point3(350, 405, 240), Point3(535, 130, 230)],
             [Point3(170, 365, 230), Point3(525, 345, 80), Point3(365, 65, 150)]]
    rasterized = [fill3D(face, SIZE) for face in faces]
    buffer = [float("inf")] * (SIZE[0] * SIZE[1])
    for points in rasterized:
        update_buffer(points, buffer, SIZE[0])
    for face, points, color in zip(faces, rasterized, (BLUE, ORANGE)):
        if key == "zbuffer":
            draw_buffer(image, points, buffer, color)
        else:
            draw_light(image, points, color, buffer, Point3(light_x, 180, -180), get_plane(*face), key == "phong")
    return image


def render_demo(key, parameters=None, source=None):
    demo = DEMO_BY_KEY[key]
    params = {parameter.key: parameter.default for parameter in demo.parameters}
    params.update(parameters or {})
    start = perf_counter()
    original = Image.new("RGB", SIZE, BACKGROUND)
    result, info = original.copy(), ""
    if key == "line":
        for i in range(int(params["rays"])):
            angle = 2 * pi * i / int(params["rays"])
            points = draw_line(320, 240, 320 + 260 * cos(angle), 240 + 205 * sin(angle))
            draw(result, points, (BLUE, GREEN, ORANGE)[i % 3])
        original = result.copy()

        result = result.crop((315, 215, 395, 275)).resize(SIZE, Image.Resampling.NEAREST)
        info = "Слева все лучи, справа увеличенный фрагмент растра."
    elif key == "circle":
        draw(result, draw_circle(320, 240, int(params["radius"])), GREEN)
        for radius in (40, 80, 120, 160, 200):
            draw(original, draw_circle(320, 240, radius), BLUE)
    elif key == "clipping":
        vertices = [(180, 100), (440, 80), (540, 260), (380, 410), (160, 350), (100, 200)]
        outline(original, vertices, ORANGE)
        result = original.copy()
        rng = np.random.default_rng(13)
        for _ in range(35):
            a, b = tuple(rng.integers((0, 0), SIZE)), tuple(rng.integers((0, 0), SIZE))
            draw(original, draw_line(*a, *b), (100, 110, 130))
            c, d = clip_segment(a, b, vertices)
            if c is not None:
                draw(result, draw_line(c.x, c.y, d.x, d.y), GREEN)
        outline(original, vertices, ORANGE)
        outline(result, vertices, ORANGE)
    elif key in ("scanline", "flood", "span", "fill_compare", "minkowski"):
        vertices = (minkowski_island(int(params["level"]), (192, 112), 256)
                    if "level" in params else [(100, 100), (530, 100), (480, 390),
                                                (350, 280), (250, 400), (120, 340)])
        original = _fill_contour(vertices)
        result = original.copy()
        if key in ("scanline", "minkowski"):
            points = scanline_fill(vertices, SIZE)
            draw(result, points, BLUE)
            outline(result, vertices, (230, 235, 245))
            info = f"Вершин: {len(vertices)}. Закрашено пикселей: {len(points)}."
        elif key == "fill_compare":
            panels, measurements = [], []
            for method, name, color in ((flood_fill, "BFS", GREEN), (span_fill, "Span", ORANGE), (None, "AEL", BLUE)):
                panel = original.copy()
                began = perf_counter()
                if method:
                    count = method(panel, (320, 240), color)
                else:
                    points = scanline_fill(vertices, SIZE)
                    draw(panel, points, color)
                    outline(panel, vertices, (230, 235, 245))
                    count = len(points)
                elapsed = (perf_counter() - began) * 1000
                panels.append(panel.resize((320, 240)))
                measurements.append(f"{name}: {elapsed:.1f} мс, {count} пикс.")
            result = Image.new("RGB", (960, 264), BACKGROUND)
            painter = ImageDraw.Draw(result)
            for i, (panel, label) in enumerate(zip(panels, ("BFS", "Span", "AEL"))):
                result.paste(panel, (i * 320, 24))
                painter.text((i * 320 + 12, 6), label, fill="white")
            info = " | ".join(measurements)
        else:
            seed = params.get("seed", (320, 240))
            count = (flood_fill if key == "flood" else span_fill)(result, seed, GREEN)
            info = f"Затравка: {tuple(seed)}. Закрашено пикселей: {count}."
    elif key == "transform2d":
        vertices = np.array([[-160, -100, 1], [110, -100, 1], [160, 70, 1], [0, 130, 1], [-130, 50, 1]])
        initial = [(p[0] + 320, p[1] + 240) for p in vertices]
        outline(original, initial, BLUE)
        transformed = scale2D(vertices, params["scale"], params["scale"])
        transformed = rotate2D(transformed, params["angle"] * pi / 180)
        transformed = shift2D(transformed, 320 + params["dx"], 240)
        outline(result, initial, (75, 85, 105))
        outline(result, [tuple(p[:2]) for p in transformed], GREEN)
    elif key == "transform3d":
        points = [Point3(x, y, z) for z in (-105, 105) for y in (-105, 105) for x in (-105, 105)]
        def wire(image, vertices, color):
            for i in range(8):
                for bit in (1, 2, 4):
                    j = i ^ bit
                    if i < j:
                        a, b = vertices[i], vertices[j]
                        draw(image, draw_line(a.x, a.y, b.x, b.y), color)
        wire(original, shift3D(rotate3D(rotate3D(points, 0.35, 3), 0.25, 2), 320, 240, 0), BLUE)
        changed = rotate3D(rotate3D(points, params["angle"] * pi / 180, 3), 0.4, 2)
        wire(result, shift3D(changed, 320, 240, 0), GREEN)
    elif key in ("zbuffer", "lambert", "phong"):
        original = _render_scene("zbuffer")
        result = _render_scene(key, params.get("light_x", 260))
        if key == "zbuffer":

            original = Image.new("RGB", SIZE, BACKGROUND)
            faces = [[Point3(125, 90, 90), Point3(350, 405, 240), Point3(535, 130, 230)],
                     [Point3(170, 365, 230), Point3(525, 345, 80), Point3(365, 65, 150)]]
            for face, color in zip(faces, (BLUE, ORANGE)):
                draw(original, [(p.x, p.y) for p in fill3D(face, SIZE)], color)
            info = "Слева порядок рисования, справа проверка глубины каждого пикселя."
    elif demo.group == "Фильтры":
        original = source.copy() if source is not None else sample_image()
        if key == "box":
            result = box_blur(original, int(params["kernel"]))
        elif key == "gaussian":
            result = gaussian_blur(original, int(params["kernel"]), float(params["sigma"]))
        elif key == "binary":
            result = binarize(original, int(params["threshold"]))
        else:
            result = {"sobel": sobel, "prewitt": prewitt, "laplacian": laplacian,
                      "bin2x2": bin_filter, "scharr": scharr}[key](original)
    elif demo.group == "Морфология":
        original = binarize(source, int(params["threshold"])) if source is not None else mask_image()
        size = int(params["kernel"])
        if key == "diff":
            result = difference(opening(original, size), closing(original, size))
        else:
            result = {"dilate": dilate, "erode": erode, "opening": opening, "closing": closing}[key](original, size)
        info = "Белый цвет — обрабатываемая область. Загруженное изображение сначала бинаризуется."
    elif demo.group == "Штрихкод EAN-13":
        original = source.copy() if source is not None and demo.accepts_image else generate(params["code"])
        if key == "ean_mask":
            result = pipeline_mask(original)
        elif key == "ean_decode":
            decoded = analyze(original)
            result, info = decoded.image, f"EAN-13: {decoded.code}. Контрольная цифра верна. Область: {decoded.box}."
        else:
            result, info = original.copy(), f"EAN-13: {normalize_code(params['code'])} · 95 модулей"
    else:
        raise ValueError(f"Неизвестная демонстрация: {key}")
    return Result(original, result, info, (perf_counter() - start) * 1000)
