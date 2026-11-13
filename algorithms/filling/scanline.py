from collections import defaultdict
from math import ceil

from algorithms.geometry import xy


def scanline_fill(vertices, size=None):
    vertices = [xy(p) for p in vertices]
    if len(vertices) < 3:
        return []
    table = defaultdict(list)
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if a[1] == b[1]:
            continue
        if a[1] > b[1]:
            a, b = b, a
        start, stop = ceil(a[1] - 0.5), ceil(b[1] - 0.5)
        if size:
            start, stop = max(0, start), min(size[1], stop)
        if start >= stop:
            continue
        slope = (b[0] - a[0]) / (b[1] - a[1])
        x = a[0] + (start + 0.5 - a[1]) * slope
        table[start].append([stop, x, slope])
    if not table:
        return []
    active, result = [], []
    last_y = max(edge[0] for edges in table.values() for edge in edges)
    for y in range(min(table), last_y):
        active = [edge for edge in active if edge[0] > y]
        active.extend(table.get(y, []))
        active.sort(key=lambda edge: edge[1])
        for left, right in zip(active[::2], active[1::2]):
            start, stop = ceil(left[1] - 0.5 - 1e-9), ceil(right[1] - 0.5 - 1e-9)
            if size:
                start, stop = max(0, start), min(size[0], stop)
            result.extend((x, y) for x in range(start, stop))
        for edge in active:
            edge[1] += edge[2]
    return result


def fill2D(vertexes):
    return scanline_fill(vertexes)
