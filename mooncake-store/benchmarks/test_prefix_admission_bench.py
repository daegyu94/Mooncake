import itertools
import unittest

from prefix_admission_bench import prefix_length


class PrefixAdmissionTest(unittest.TestCase):
    def test_all_holes_and_rank_failures(self):
        blocks = [[f"{i}/{rank}" for rank in range(2)] for i in range(4)]
        keys = [key for block in blocks for key in block]
        for states in itertools.product([0, 1, -1], repeat=8):
            values = dict(zip(keys, states))

            def query(batch, values=values):
                return [values[k] for k in batch]

            expected = next(
                (
                    i
                    for i, block in enumerate(blocks)
                    if any(values[k] != 1 for k in block)
                ),
                4,
            )
            for probe in [0, 1, 2, 4, 8]:
                self.assertEqual(prefix_length(query, blocks, probe), expected)

    def test_no_cached_policy_state(self):
        blocks = [["policy0/a"], ["policy0/b"]]
        self.assertEqual(prefix_length(lambda keys: [1] * len(keys), blocks, 1), 2)
        self.assertEqual(prefix_length(lambda keys: [0] * len(keys), blocks, 1), 0)

    def test_empty_and_malformed(self):
        self.assertEqual(prefix_length(lambda _: self.fail(), [], 16), 0)
        with self.assertRaises(ValueError):
            prefix_length(lambda _: [], [["a"]], 16)
        with self.assertRaises(ValueError):
            prefix_length(lambda _: [], [["a"]], -1)
        with self.assertRaises(ValueError):
            prefix_length(lambda _: [], [[]], 1)


if __name__ == "__main__":
    unittest.main()
