def draw_line(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
    steep = abs(y2 - y1) > abs(x2 - x1)
    if steep:
        x1, y1, x2, y2 = y1, x1, y2, x2
    reverse = x1 > x2
    if reverse:
        x1, x2, y1, y2 = x2, x1, y2, y1
    dx, dy = x2 - x1, abs(y2 - y1)
    error, y = dx // 2, y1
    step = 1 if y1 < y2 else -1
    points = []
    for x in range(x1, x2 + 1):
        points.append((y, x) if steep else (x, y))
        error -= dy
        if error < 0:
            y += step
            error += dx
    return points[::-1] if reverse else points


def draw_circle(center_x, center_y, radius):
    if radius < 0 or int(radius) != radius:
        raise ValueError("Радиус должен быть целым неотрицательным числом")
    cx, cy, radius = int(round(center_x)), int(round(center_y)), int(radius)
    x, y, decision = radius, 0, 1 - radius
    points = set()
    while x >= y:
        for dx, dy in ((x, y), (-x, y), (x, -y), (-x, -y),
                       (y, x), (-y, x), (y, -x), (-y, -x)):
            points.add((cx + dx, cy + dy))
        y += 1
        if decision < 0:
            decision += 2 * y + 1
        else:
            x -= 1
            decision += 2 * (y - x) + 1
    return sorted(points)


def draw(image, points, color):
    width, height = image.size
    pixels = image.load()
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < width and 0 <= y < height:
            pixels[x, y] = color
