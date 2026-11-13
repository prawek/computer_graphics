from algorithms.filling import scanline_fill
from algorithms.geometry import Point3


def get_plane(point1, point2, point3):
    u = (point2.x - point1.x, point2.y - point1.y, point2.z - point1.z)
    v = (point3.x - point1.x, point3.y - point1.y, point3.z - point1.z)
    a, b, c = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
               u[0] * v[1] - u[1] * v[0])
    return a, b, c, -(a * point1.x + b * point1.y + c * point1.z)


def fill3D(vertexes, size=None):
    if len(vertexes) < 3:
        return []
    a, b, c, d = get_plane(*vertexes[:3])
    if abs(c) < 1e-12:
        return []
    return [Point3(x, y, -(d + a * (x + 0.5) + b * (y + 0.5)) / c)
            for x, y in scanline_fill(vertexes, size)]


def update_buffer(points, z_buffer, width=800):
    if width <= 0 or len(z_buffer) % width:
        raise ValueError("Размер Z-буфера должен быть кратен положительной ширине")
    height = len(z_buffer) // width
    for p in points:
        x, y = int(p.x), int(p.y)
        if 0 <= x < width and 0 <= y < height:
            index = y * width + x
            z_buffer[index] = min(z_buffer[index], p.z)


def draw_buffer(image, points, z_buffer, color):
    if len(z_buffer) != image.width * image.height:
        raise ValueError("Размер Z-буфера не совпадает с изображением")
    for p in points:
        x, y = int(p.x), int(p.y)
        if 0 <= x < image.width and 0 <= y < image.height:
            if p.z <= z_buffer[y * image.width + x] + 1e-9:
                image.putpixel((x, y), color)
