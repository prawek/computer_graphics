import math
import unittest

import numpy as np
from PIL import Image

from algorithms.barcode import analyze, checksum, decode_bits, encode_bits, generate
from algorithms.clipping import Polygon, clip_segment
from algorithms.filling import flood_fill, scanline_fill, span_fill
from algorithms.filters import (binarize, bin_filter, box_blur, gaussian_blur,
                                laplacian, prewitt, scharr, sobel, sobel_filter)
from algorithms.fractals import minkowski_island
from algorithms.geometry import Point2, Point3
from algorithms.morphology import closing, difference, dilate, erode, opening
from algorithms.rasterization import draw, draw_circle, draw_line
from algorithms.rendering import (diffuse_light, draw_buffer, fill3D, get_plane,
                                  specular_light, update_buffer)
from algorithms.transformations import (reflect2D, reflect3D, rotate2D, rotate3D,
                                       scale2D, scale3D, shift2D, shift3D)


class RasterTests(unittest.TestCase):
    def test_line_all_octants_endpoints_connectivity_and_reverse(self):
        for x in range(-7, 8):
            for y in range(-7, 8):
                points = draw_line(0, 0, x, y)
                self.assertEqual(points[0], (0, 0))
                self.assertEqual(points[-1], (x, y))
                self.assertEqual(len(points), max(abs(x), abs(y)) + 1)
                self.assertEqual(points[::-1], draw_line(x, y, 0, 0))
                for a, b in zip(points, points[1:]):
                    self.assertEqual(max(abs(a[0] - b[0]), abs(a[1] - b[1])), 1)

    def test_circle_symmetry_and_radius(self):
        for radius in range(21):
            points = set(draw_circle(0, 0, radius))
            for x, y in points:
                self.assertIn((-x, y), points)
                self.assertIn((y, x), points)
                self.assertLessEqual(abs(math.hypot(x, y) - radius), 0.6)
        with self.assertRaises(ValueError):
            draw_circle(0, 0, -1)

    def test_safe_drawing(self):
        image = Image.new("RGB", (3, 2))
        draw(image, [(-1, 0), (0, 0), (3, 0), (1, 4)], (255, 0, 0))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))


class ClipTests(unittest.TestCase):
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_both_orientations_inside_crossing_parallel_and_touch(self):
        cases = [((2, 2), (8, 8), ((2, 2), (8, 8))),
                 ((-5, 5), (15, 5), ((0, 5), (10, 5))),
                 ((-5, 0), (15, 0), ((0, 0), (10, 0))),
                 ((-5, -5), (0, 0), ((0, 0), (0, 0))),
                 ((5, 5), (5, 5), ((5, 5), (5, 5))),
                 ((-5, -1), (15, -1), None), ((11, 5), (11, 5), None)]
        for vertices in (self.square, self.square[::-1]):
            for a, b, expected in cases:
                with self.subTest(vertices=vertices, a=a, b=b):
                    result = clip_segment(a, b, vertices)
                    if expected is None:
                        self.assertEqual(result, (None, None))
                    else:
                        np.testing.assert_allclose([(p.x, p.y) for p in result], expected)

    def test_polygon_uses_its_own_vertices(self):
        first = Polygon([Point2(*p) for p in self.square])
        second = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
        self.assertIsNotNone(first.cyruse_beck((2, 2), (8, 8))[0])
        self.assertEqual(second.cyruse_beck((2, 2), (8, 8)), (None, None))

    def test_reject_concave_and_degenerate(self):
        for vertices in ([(0, 0), (5, 2), (10, 0), (10, 10), (0, 10)], [(0, 0), (1, 1), (2, 2)]):
            with self.assertRaises(ValueError):
                clip_segment((0, 0), (10, 10), vertices)


class FillTests(unittest.TestCase):
    def test_scanline_square_and_orientation(self):
        polygon = [(1, 1), (5, 1), (5, 5), (1, 5)]
        expected = {(x, y) for x in range(1, 5) for y in range(1, 5)}
        self.assertEqual(set(scanline_fill(polygon)), expected)
        self.assertEqual(set(scanline_fill(polygon[::-1])), expected)
        self.assertEqual(set(scanline_fill(polygon, (3, 3))), {(1, 1), (1, 2), (2, 1), (2, 2)})

    def test_scanline_concavity_and_fractional_vertices(self):
        polygon = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        points = set(scanline_fill(polygon))
        self.assertEqual(len(points), 12)
        self.assertNotIn((3, 3), points)
        self.assertEqual(len(scanline_fill([(0.2, 0.2), (2.2, 0.2), (2.2, 2.2), (0.2, 2.2)])), 4)

    def test_flood_and_span_agree_on_random_regions(self):
        rng = np.random.default_rng(16)
        for _ in range(12):
            array = (rng.random((31, 43)) > 0.3).astype(np.uint8) * 255
            first, second = Image.fromarray(array), Image.fromarray(array)
            count1, count2 = flood_fill(first, (21, 15), 128), span_fill(second, (21, 15), 128)
            self.assertEqual(count1, count2)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(count1, np.count_nonzero(np.asarray(first) == 128))

    def test_noop_same_color_and_outside_seed(self):
        for function in (flood_fill, span_fill):
            image = Image.new("L", (5, 7), 0)
            self.assertEqual(function(image, (0, 0), 0), 0)
            self.assertEqual(function(image, (-1, 0), 255), 0)
            self.assertEqual(function(image, (0, 0), 255), 35)


class FilterTests(unittest.TestCase):
    def test_constant_image_and_alpha(self):
        image = Image.new("RGBA", (8, 9), (90, 100, 110, 73))
        for function in (box_blur, gaussian_blur):
            np.testing.assert_array_equal(function(image), image)
        for function in (sobel, prewitt, scharr, laplacian, bin_filter):
            array = np.asarray(function(image))
            self.assertTrue(np.all(array[..., :3] == 0))
            self.assertTrue(np.all(array[..., 3] == 73))

    def test_sobel_two_dimensional_neighborhood(self):

        step = np.zeros((7, 9), dtype=np.uint8)
        step[:, 4:] = 30
        expected = np.zeros_like(step)
        expected[:, 3:5] = 120
        np.testing.assert_array_equal(sobel(Image.fromarray(step)), expected)
        np.testing.assert_array_equal(sobel(Image.fromarray(step.T)), expected.T)
        np.testing.assert_array_equal(np.array(sobel_filter(step.ravel(), 9, 7)).reshape(7, 9), expected)

    def test_gaussian_impulse_symmetry_and_normalization(self):
        impulse = np.zeros((11, 11), dtype=np.uint8)
        impulse[5, 5] = 255
        blurred = np.asarray(gaussian_blur(Image.fromarray(impulse), 5, 1))
        np.testing.assert_array_equal(blurred, blurred[::-1])
        np.testing.assert_array_equal(blurred, blurred.T)
        self.assertLessEqual(abs(int(blurred.sum()) - 255), 8)

    def test_invalid_parameters_and_binary(self):
        image = Image.new("L", (2, 3), 128)
        for kernel in (0, 2, -3):
            with self.assertRaises(ValueError):
                box_blur(image, kernel)
        with self.assertRaises(ValueError):
            gaussian_blur(image, 3, 0)
        with self.assertRaises(ValueError):
            sobel_filter([0, 0], 3, 3)
        self.assertEqual(set(np.asarray(binarize(image, 127)).ravel()), {255})
        self.assertEqual(set(np.asarray(binarize(image, 128)).ravel()), {0})


class MorphologyTests(unittest.TestCase):
    def test_single_pixel_and_duality(self):
        array = np.zeros((9, 9), dtype=np.uint8)
        array[4, 4] = 255
        image = Image.fromarray(array)
        self.assertEqual(np.count_nonzero(dilate(image)), 9)
        self.assertEqual(np.count_nonzero(erode(image)), 0)
        np.testing.assert_array_equal(opening(image), np.zeros_like(array))
        np.testing.assert_array_equal(closing(image), array)
        inverse = Image.fromarray(255 - array)
        np.testing.assert_array_equal(np.asarray(dilate(image)), 255 - np.asarray(erode(inverse)))

    def test_xor_and_size_validation(self):
        first, second = Image.new("L", (2, 3), 0), Image.new("L", (2, 3), 255)
        np.testing.assert_array_equal(difference(first, second), second)
        with self.assertRaises(ValueError):
            difference(first, Image.new("L", (3, 2)))


class TransformTests(unittest.TestCase):
    def test_2d_operations_and_reflection(self):
        points = [[2, 3, 1]]
        np.testing.assert_allclose(rotate2D(points, math.pi / 2), [[-3, 2, 1]], atol=1e-12)
        np.testing.assert_allclose(shift2D(points, 4, 5), [[6, 8, 1]])
        np.testing.assert_allclose(scale2D(points, 2, 4), [[4, 12, 1]])
        np.testing.assert_allclose(reflect2D(points, 10, 1), [[2, 17, 1]])
        np.testing.assert_allclose(reflect2D(points, 10, 0), [[18, 3, 1]])

    def test_3d_no_early_rounding_and_right_handed_rotation(self):
        points = [Point3(1, 0, 0)]
        rotated = rotate3D(points, math.pi / 2, 3)[0]
        np.testing.assert_allclose((rotated.x, rotated.y, rotated.z), (0, 0, -1), atol=1e-12)
        scaled = scale3D(points, 0.25, 1, 1)[0]
        self.assertEqual(scaled.x, 0.25)
        self.assertEqual(shift3D(points, 2, 3, 4)[0], Point3(3, 3, 4))
        self.assertEqual(reflect3D(points, 2, 2)[0], Point3(3, 0, 0))


class RenderTests(unittest.TestCase):
    def test_zbuffer_arbitrary_size_and_order_independence(self):
        near = fill3D([Point3(1, 1, 2), Point3(7, 1, 2), Point3(1, 7, 2)], (9, 8))
        far = fill3D([Point3(1, 1, 5), Point3(7, 1, 5), Point3(1, 7, 5)], (9, 8))
        results = []
        for faces in ([(near, (255, 0, 0)), (far, (0, 0, 255))], [(far, (0, 0, 255)), (near, (255, 0, 0))]):
            buffer = [float("inf")] * 72
            image = Image.new("RGB", (9, 8))
            for points, _ in faces:
                update_buffer(points, buffer, 9)
            for points, color in faces:
                draw_buffer(image, points, buffer, color)
            results.append(np.asarray(image))
            self.assertEqual(image.getpixel((2, 2)), (255, 0, 0))
        np.testing.assert_array_equal(*results)
        self.assertEqual(fill3D([Point3(1, 1, 0), Point3(1, 4, 0), Point3(1, 4, 5)]), [])

    def test_lighting_degenerate_and_back_light(self):
        point = Point3(0, 0, 0)
        self.assertEqual(diffuse_light(point, point, (255, 0, 0), 0, 0, 0), Point3(0, 0, 0))
        self.assertEqual(diffuse_light(Point3(0, 0, -1), point, (255, 0, 0), 0, 0, 1), Point3(0, 0, 0))
        self.assertEqual(specular_light(Point3(0, 0, 3), (0, 0, 1), Point3(0, 0, 2), point, (255, 255, 255)), Point3(255, 255, 255))


class BarcodeTests(unittest.TestCase):
    def test_known_code_and_all_first_digit_parities(self):
        self.assertEqual(checksum("590123412345"), "7")
        for first in "0123456789":
            code = first + "90123412345"
            expected = code + checksum(code)
            self.assertEqual(decode_bits(encode_bits(code)), expected)
            self.assertEqual(len(encode_bits(code)), 95)

    def test_generate_and_read_including_reverse_and_embedded(self):
        for module_width in (1, 2, 4):
            image = generate("590123412345", module_width, 75, 20)
            self.assertEqual(analyze(image).code, "5901234123457")
            self.assertEqual(analyze(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).code, "5901234123457")
        canvas = Image.new("RGB", (650, 400), "white")
        canvas.paste(generate("400638133393", 3, 80, 15), (140, 210))
        self.assertEqual(analyze(canvas).code, "4006381333931")

    def test_reject_invalid_data_and_checksum(self):
        for value in ("123", "5901234123458", "a90123412345", "５90123412345"):
            with self.assertRaises(ValueError):
                generate(value)
        with self.assertRaises(ValueError):
            analyze(Image.new("RGB", (100, 100), "white"))


class FractalTests(unittest.TestCase):
    def test_edge_growth_closure_and_area(self):
        for level in range(4):
            vertices = minkowski_island(level, side=128)
            self.assertEqual(len(vertices), 4 * 8**level)
            area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(vertices, vertices[1:] + vertices[:1])) / 2
            self.assertAlmostEqual(area, 128**2)


if __name__ == "__main__":
    unittest.main()
