import random
from math import log2, ceil
from typing import Callable, Sized
from bitarray import bitarray
import numpy as np

w = 64
p = (1 << 61) - 1


def get_rand_odd(k: int) -> int:
    """

    :param k: the number of random bits
    :return: a python integer with k random bits
    """
    x = random.getrandbits(k)
    while not (x & 1):
        x = random.getrandbits(k)
    return x


def hash_shift(hash_x, a, b):
    return (a * hash_x + b) & (1 << w) - 1


def get_hash_functions_shift(n: int) -> Sized[Callable]:
    ab_pairs = set()
    while len(ab_pairs) != n:
        a = get_rand_odd(w)  # uniformly random odd w-bit integer a
        b = random.randint(0, p)
        ab_pairs.add((a, b))

    def wrapper(_a, _b) -> Callable:
        return lambda item: hash_shift(hash(item), _a, _b)

    return tuple(wrapper(a, b) for a, b in ab_pairs)


class BloomFilter:
    """ Goal : Design our design structure so that
                - if x ∈ S
                - if x ∉ S we return false with a probability of (1 - ε)
                - the amount of space depends only on the size of the universe of n and ε and not the size of the universe 𝒰
    """
    MIN_ERR = 0.000001

    def __new__(cls, expected_number_of_items: int, error_rate: float):
        """
        :param expected_number_of_items:
        :param error_rate: The probability of false positives
        """

        if error_rate < BloomFilter.MIN_ERR or error_rate >= 1.0:
            raise ValueError(f"The error rate must be between {BloomFilter.MIN_ERR} and {1.0}")

        self = super(BloomFilter, cls).__new__(cls)

        c = 1.4426950408889634  # log_2(e)
        d = 0.6931471805599453  # log_e(2)
        c_log_epsilon = c * log2(1.0 / error_rate)
        self.n = expected_number_of_items
        self.m = ceil(c_log_epsilon * expected_number_of_items)
        self.k = ceil(c_log_epsilon * d)
        self.bloom = bitarray(self.m)
        self.bloom.setall(0)
        self.hashes = get_hash_functions_shift(self.k)
        self.seen = 0
        return self

    def offset(self, hash_val):
        return (hash_val & p) % self.m

    def add(self, item):
        for hash_function in self.hashes:
            self.bloom[self.offset(hash_function(item))] = 1
        self.seen += 1

    def addall(self, items):
        for item in items:
            self.add(item)

    def __contains__(self, item):
        for hash_function in self.hashes:
            if not self.bloom[self.offset(hash_function(item))]:
                return False
        return True

    def theoretical_error_rate(self):
        err = (1 - 1 / self.m) ** (self.k * self.seen)
        return f"{err * 100:0.4f} %"

    def false_positive_probability(self):
        return (1 - (1 - 1 / self.m) ** (self.k * self.seen)) ** self.k

    def __repr__(self):
        return repr(self.bloom)


def test_insertion_100(n, error_rate):
    bf = BloomFilter(n, error_rate)
    A = set(map(tuple, np.random.randint(0, 256, (n, 4))))
    B = set(map(tuple, np.random.randint(0, 256, (n, 4)))) - A
    print(len(A), len(B))
    bf.addall(A)
    fp = sum(x in bf for x in B)
    acc = 1 - fp / len(B)
    print('{} hashes, {} false positives, {:.4f} accuracy'.format(bf.k, fp, acc))


if __name__ == '__main__':
    test_insertion_100(100000, 0.01)

    # bf = BloomFilter(3, 0.1)
    # bf.add("Erastus")
    # bf.add("Murungi")
    # bf.add(10000)
    # print(bf.seen)
    # print(bf.theoretical_error_rate())
    # print(bf.false_positive_probability())
    # print(10000 in bf)
