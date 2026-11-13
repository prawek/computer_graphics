def minkowski_island(iterations=2, origin=(0, 0), side=256):
    if not isinstance(iterations, int) or not 0 <= iterations <= 5:
        raise ValueError("Число итераций должно быть целым от 0 до 5")
    if side <= 0:
        raise ValueError("Сторона должна быть положительной")
    x, y = origin
    vertices = [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]
    pattern = [(0, 0), (1, 0), (1, -1), (2, -1), (2, 0), (2, 1), (3, 1), (3, 0)]
    for _ in range(iterations):
        result = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            ux, uy = (b[0] - a[0]) / 4, (b[1] - a[1]) / 4
            vx, vy = -uy, ux
            result.extend((a[0] + i * ux + j * vx, a[1] + i * uy + j * vy) for i, j in pattern)
        vertices = result
    return vertices
