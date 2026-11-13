import numpy as np

from algorithms.geometry import Point3


def _apply2(points, matrix):
    return [np.asarray(point, dtype=float) @ np.asarray(matrix, dtype=float).T for point in points]


def rotate2D(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return _apply2(points, [[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shift2D(points, dx, dy):
    return _apply2(points, [[1, 0, dx], [0, 1, dy], [0, 0, 1]])


def scale2D(points, sx, sy):
    return _apply2(points, [[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def reflect2D(points, line, axis=1):
    if axis not in (0, 1):
        raise ValueError("Ось 2D: 0 (x) или 1 (y)")
    matrix = np.eye(3)
    matrix[axis, axis], matrix[axis, 2] = -1, 2 * line
    return _apply2(points, matrix)


def to_numpy(points):
    return [np.array([[p.x, p.y, p.z, 1.0]]) for p in points]


def to_point3(points):
    return [Point3(*np.asarray(p).reshape(-1)[:3]) for p in points]


def _apply3(points, matrix):
    return to_point3([p @ np.asarray(matrix, dtype=float).T for p in to_numpy(points)])


def rotate3D(points, angle, axis=1):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 1:
        matrix = [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif axis == 2:
        matrix = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
    elif axis == 3:
        matrix = [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
    else:
        raise ValueError("Ось 3D: 1 (z), 2 (x) или 3 (y)")
    return _apply3(points, matrix)


def shift3D(points, dx, dy, dz):
    matrix = np.eye(4)
    matrix[:3, 3] = dx, dy, dz
    return _apply3(points, matrix)


def scale3D(points, sx, sy, sz):
    return _apply3(points, np.diag([sx, sy, sz, 1]))


def reflect3D(points, line, axis=1):
    if axis not in (1, 2, 3):
        raise ValueError("Ось 3D: 1 (z), 2 (x) или 3 (y)")
    index = {1: 2, 2: 0, 3: 1}[axis]
    matrix = np.eye(4)
    matrix[index, index], matrix[index, 3] = -1, 2 * line
    return _apply3(points, matrix)
