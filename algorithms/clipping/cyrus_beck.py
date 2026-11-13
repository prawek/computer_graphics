from algorithms.geometry import Point2, xy

EPS = 1e-9


def _convex_vertices(vertices):
    vertices = [xy(p) for p in vertices]
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3:
        raise ValueError("Нужны хотя бы три вершины выпуклого многоугольника")
    area = sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(vertices, vertices[1:] + vertices[:1]))
    if abs(area) <= EPS:
        raise ValueError("Многоугольник вырожден")
    orientation = 1 if area > 0 else -1
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if a == b:
            raise ValueError("Соседние вершины совпадают")
        for p in vertices:
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if orientation * cross < -EPS:
                raise ValueError("Кирус — Бек требует выпуклый многоугольник")
    return vertices, orientation


def clip_segment(a, b, vertices):
    vertices, orientation = _convex_vertices(vertices)
    ax, ay = xy(a)
    bx, by = xy(b)
    dx, dy = bx - ax, by - ay
    begin, end = 0.0, 1.0
    for p, q in zip(vertices, vertices[1:] + vertices[:1]):
        nx, ny = orientation * (p[1] - q[1]), orientation * (q[0] - p[0])
        distance = nx * (ax - p[0]) + ny * (ay - p[1])
        rate = nx * dx + ny * dy
        if abs(rate) <= EPS:
            if distance < -EPS:
                return None, None
            continue
        t = -distance / rate
        if rate > 0:
            begin = max(begin, t)
        else:
            end = min(end, t)
        if begin > end + EPS:
            return None, None
    if begin > end:
        begin = end = (begin + end) / 2
    return Point2(ax + begin * dx, ay + begin * dy), Point2(ax + end * dx, ay + end * dy)


class Polygon:
    def __init__(self, vertexes):
        self.vertexes = list(vertexes)
        _convex_vertices(self.vertexes)

    def cyruse_beck(self, a, b):
        return clip_segment(a, b, self.vertexes)
