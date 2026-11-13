from collections import deque


def _prepare(image, seed, color):
    x, y = map(int, seed)
    if not (0 <= x < image.width and 0 <= y < image.height):
        return None
    target = image.getpixel((x, y))

    probe = image.crop((0, 0, 1, 1))
    probe.putpixel((0, 0), color)
    replacement = probe.getpixel((0, 0))
    if target == replacement:
        return None


    image.putpixel((x, y), target)
    return x, y, image.load(), target, replacement


def flood_fill(image, seed, color):
    prepared = _prepare(image, seed, color)
    if prepared is None:
        return 0
    x, y, pixels, target, color = prepared
    queue = deque([(x, y)])
    pixels[x, y] = color
    count = 1
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < image.width and 0 <= ny < image.height and pixels[nx, ny] == target:
                pixels[nx, ny] = color
                count += 1
                queue.append((nx, ny))
    return count


def span_fill(image, seed, color):
    prepared = _prepare(image, seed, color)
    if prepared is None:
        return 0
    x, y, pixels, target, color = prepared
    stack, count = [(x, y)], 0
    while stack:
        x, y = stack.pop()
        if pixels[x, y] != target:
            continue
        left = right = x
        while left > 0 and pixels[left - 1, y] == target:
            left -= 1
        while right + 1 < image.width and pixels[right + 1, y] == target:
            right += 1
        for px in range(left, right + 1):
            pixels[px, y] = color
        count += right - left + 1
        for ny in (y - 1, y + 1):
            if not 0 <= ny < image.height:
                continue
            inside = False
            for px in range(left, right + 1):
                matches = pixels[px, ny] == target
                if matches and not inside:
                    stack.append((px, ny))
                inside = matches
    return count
