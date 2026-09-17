"""Explicit ERI derivative oracle, confined to small-system tests."""

import itertools
import numpy as np


# Eight chemists' ERI symmetries, used to place differentiated AO slots.
_ERI_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _eri_first(ip1, mask, xyz):
    result = np.zeros_like(ip1[0])
    for slot in range(4):
        permutation = next(p for p in _ERI_PERMUTATIONS if p[slot] == 0)
        shape = [1] * 4
        shape[slot] = len(mask)
        # Nuclear displacement is minus the AO electronic-coordinate derivative.
        result -= ip1[xyz].transpose(permutation) * mask.reshape(shape)
    return result


def _eri_second(ipip1, ipvip1, ip1ip2, mask_a, mask_b, xyz_a, xyz_b):
    result = np.zeros_like(ipip1[0, 0])
    for first, second in itertools.product(range(4), repeat=2):
        if first == second:
            primitive = ipip1
            permutation = next(p for p in _ERI_PERMUTATIONS if p[first] == 0)
        else:
            partner = 1 if first // 2 == second // 2 else 2
            primitive = ipvip1 if partner == 1 else ip1ip2
            permutation = next(
                p for p in _ERI_PERMUTATIONS if p[first] == 0 and p[second] == partner
            )
        shape_a = [1] * 4
        shape_b = [1] * 4
        shape_a[first] = len(mask_a)
        shape_b[second] = len(mask_b)
        result += (primitive[xyz_a, xyz_b].transpose(permutation)
                   * mask_a.reshape(shape_a) * mask_b.reshape(shape_b))
    return result
