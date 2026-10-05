"""CPU checks for deterministic per-image statistics, not augmentation equivalence."""

import copy
import math
import random
import unittest

from pipeline import normalize


class NormalizeTests(unittest.TestCase):
    def test_known_population_statistics(self):
        result = normalize([[1, 2, 3], [10, 14]])
        expected = [
            [-1 / math.sqrt(2 / 3 + 1e-5), 0.0, 1 / math.sqrt(2 / 3 + 1e-5)],
            [-2 / math.sqrt(4 + 1e-5), 2 / math.sqrt(4 + 1e-5)],
        ]
        for actual_image, expected_image in zip(result, expected):
            for actual, reference in zip(actual_image, expected_image):
                self.assertAlmostEqual(actual, reference, places=12)

    def test_batch_independence(self):
        images = [[0, 10, 250], [-5, 3], [100] * 4]
        unrelated = [1_000_000, -2_000_000, 50]
        baseline = normalize(images)
        self.assertEqual(normalize(images + [unrelated])[:-1], baseline)
        self.assertEqual(normalize([unrelated] + images)[1:], baseline)
        self.assertEqual(normalize(list(reversed(images))), list(reversed(baseline)))
        self.assertEqual(baseline, [normalize([image])[0] for image in images])

    def test_shape_and_input_preservation(self):
        images = [[1, 2, 3], [4], [5, 7]]
        before = copy.deepcopy(images)
        result = normalize(images)
        self.assertEqual(images, before)
        self.assertIsInstance(result, list)
        self.assertEqual([len(image) for image in result], [3, 1, 2])
        self.assertIsNot(result, images)
        for source, output in zip(images, result):
            self.assertIsInstance(output, list)
            self.assertIsNot(source, output)

    def test_constant_and_singleton_images(self):
        self.assertEqual(normalize([[8, 8, 8], [-9]]), [[0.0, 0.0, 0.0], [0.0]])

    def test_empty_batch(self):
        self.assertEqual(normalize([]), [])

    def test_empty_image_rejected_without_input_changes(self):
        for images in ([[]], [[], [1, 2]], [[1, 2], []]):
            with self.subTest(images=images):
                before = copy.deepcopy(images)
                with self.assertRaises(ValueError):
                    normalize(images)
                self.assertEqual(images, before)

    def test_seeded_finite_data_statistics(self):
        rng = random.Random(1742)
        for size in (2, 3, 7, 31, 100):
            for _ in range(20):
                image = [rng.uniform(-255, 255) for _ in range(size)]
                output = normalize([image])[0]
                mean = math.fsum(image) / size
                variance = math.fsum((value - mean) ** 2 for value in image) / size
                self.assertAlmostEqual(math.fsum(output) / size, 0.0, places=12)
                self.assertAlmostEqual(
                    math.fsum(value * value for value in output) / size,
                    variance / (variance + 1e-5),
                    places=12,
                )
                for source, actual in zip(image, output):
                    reference = (source - mean) / math.sqrt(variance + 1e-5)
                    self.assertAlmostEqual(actual, reference, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
