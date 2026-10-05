import copy
import unittest

import numpy as np

from pipeline import convert_qkv


class ConvertQKVTests(unittest.TestCase):
    def test_cpu_projection_matches_component_outputs(self):
        # Non-square weights and a batched input exercise projection behavior.
        inputs = np.array([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, 1.5]])
        for heads in (1, 2, 3):
            with self.subTest(heads=heads):
                source = (np.arange(3 * heads * 4).reshape(3 * heads, 4) - 7.0) / 4.0
                converted = np.array(convert_qkv(source.tolist(), heads))
                actual = inputs @ converted.T
                # Each stride selects one component from the source layout.
                expected = np.concatenate(
                    [inputs @ source[component::3].T for component in range(3)],
                    axis=1,
                )
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
                if heads > 1:
                    self.assertFalse(np.array_equal(inputs @ source.T, expected))

    def test_row_order_and_independent_copies(self):
        source = [[i, i + 10] for i in range(6)]
        before = copy.deepcopy(source)
        result = convert_qkv(source, 2)
        self.assertEqual(result, [source[i] for i in (0, 3, 1, 4, 2, 5)])
        self.assertEqual(source, before)
        for row in result:
            self.assertTrue(all(row is not original for original in source))
        result[0][0] = -999
        self.assertEqual(source, before)

    def test_invalid_layouts_raise_value_error(self):
        invalid = [
            ([[1]] * 3, 0),
            ([[1]] * 3, -1),
            ([[1]] * 3, 1.5),
            ([[1]] * 3, "1"),
            ([[1]] * 3, True),
            ([], 1),
            ([[1], [2]], 1),
            ([[1], [2], [3], [4]], 1),
            ([[], [], []], 1),
            ([[1], [2, 3], [4]], 1),
            ([[1], None, [3]], 1),
            (None, 1),
        ]
        for weight, heads in invalid:
            with self.subTest(weight=weight, heads=heads):
                with self.assertRaises(ValueError):
                    convert_qkv(weight, heads)


if __name__ == "__main__":
    unittest.main()
