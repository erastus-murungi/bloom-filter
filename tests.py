import unittest
import numpy as np

from bloom import BloomFilter


class TestBloomFilter(unittest.TestCase):
    def test_insertion_100(self):
        n = 100000
        bf = BloomFilter(n, 0.5464)
        n = 10 ** 6
        A = set(map(tuple, np.random.randint(0, 256, (n, 4))))
        B = set(map(tuple, np.random.randint(0, 256, (n, 4)))) - A
        print(len(A), len(B))
        bf.addall(A)
        fp = sum(x in bf for x in B)
        acc = 1 - fp / len(B)
        print('{} hashes, {} false positives, {:.4f} accuracy'.format(bf.k, fp, acc))





