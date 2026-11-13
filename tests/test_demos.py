import unittest

from algorithms.demos import DEMOS, render_demo


class DemoTests(unittest.TestCase):
    def test_every_demo_produces_images_without_external_files(self):
        for demo in DEMOS:
            with self.subTest(demo=demo.key):
                result = render_demo(demo.key)
                self.assertGreater(result.original.width, 0)
                self.assertGreater(result.processed.height, 0)
                self.assertGreaterEqual(result.elapsed_ms, 0)


if __name__ == "__main__":
    unittest.main()
