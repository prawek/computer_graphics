from math import sqrt

from algorithms.geometry import Point3


def _unit(vector):
    length = sqrt(sum(v * v for v in vector))
    return tuple(v / length for v in vector) if length > 1e-12 else (0.0, 0.0, 0.0)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _direction(a, b):
    return _unit((a.x - b.x, a.y - b.y, a.z - b.z))


def diffuse_light(light_point, point, color, A, B, C):
    strength = max(0.0, _dot(_direction(light_point, point), _unit((A, B, C))))
    return Point3(*(strength * channel for channel in color))


def specular_light(view_point, normal, light_point, point, color, shininess=32):
    n, light = _unit(normal[:3]), _direction(light_point, point)
    incidence = _dot(n, light)
    if incidence <= 0:
        return Point3(0, 0, 0)
    reflection = tuple(2 * incidence * ni - li for ni, li in zip(n, light))
    strength = max(0.0, _dot(reflection, _direction(view_point, point))) ** shininess
    return Point3(*(strength * channel for channel in color))


def draw_light(image, points, color, z_buffer, light_point, norm, specular=True):
    if len(z_buffer) != image.width * image.height:
        raise ValueError("Размер Z-буфера не совпадает с изображением")
    view = Point3(image.width / 2, image.height / 2, -500)
    normal = norm[:3]
    if normal[2] > 0:
        normal = tuple(-n for n in normal)
    for p in points:
        x, y = int(p.x), int(p.y)
        if not (0 <= x < image.width and 0 <= y < image.height):
            continue
        if p.z > z_buffer[y * image.width + x] + 1e-9:
            continue
        diffuse = diffuse_light(light_point, p, color, *normal)
        shine = specular_light(view, normal, light_point, p, (255, 255, 255)) if specular else Point3(0, 0, 0)
        channels = zip(color, (diffuse.x, diffuse.y, diffuse.z), (shine.x, shine.y, shine.z))
        image.putpixel((x, y), tuple(max(0, min(255, round(0.12 * base + 0.8 * d + 0.5 * s)))
                                   for base, d, s in channels))
