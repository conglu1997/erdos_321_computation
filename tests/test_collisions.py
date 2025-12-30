import unittest

from collisions import find_relation_grouped


class CollisionGroupTests(unittest.TestCase):
    def test_find_relation_grouped_respects_group_signs(self):
        rel = find_relation_grouped([1, 2, 3, 6], [[2, 3, 6]])
        self.assertIsNotNone(rel)
        # The grouped elements should appear together on one side of the relation.
        self.assertEqual(set(rel.plus), {2, 3, 6})
        self.assertEqual(set(rel.minus), {1})


if __name__ == "__main__":
    unittest.main()
