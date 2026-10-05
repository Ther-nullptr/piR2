"""Protect interval accounting against gaps, duplicate kernels, and touching edges."""

import unittest

from coexecution.analyze_trace import intersect, merge


class TestTimeline(unittest.TestCase):
    def test_overlap_is_kernel_time_not_enclosing_span(self):
        first = merge([(0, 2), (8, 10)])
        second = merge([(3, 7)])
        self.assertEqual(intersect(first, second), [])

    def test_duplicate_and_touching_intervals_are_not_double_counted(self):
        self.assertEqual(merge([(0, 4), (1, 3), (4, 7), (9, 10)]), [(0, 7), (9, 10)])
        self.assertEqual(intersect([(0, 7), (9, 10)], [(3, 9)]), [(3, 7)])


if __name__ == "__main__":
    unittest.main()
