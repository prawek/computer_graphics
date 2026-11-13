from dataclasses import dataclass


@dataclass
class Point2:
    x: float
    y: float


@dataclass
class Point3:
    x: float
    y: float
    z: float


def xy(point):
    return (point.x, point.y) if hasattr(point, "x") else tuple(point[:2])
