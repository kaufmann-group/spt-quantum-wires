import numpy as np
import qutip as qt

from .base import SPTChain


class Spin1AKLT(SPTChain):
    bulk_spin = 1

    def tensors(self):
        # Cartesian basis of Section III C, normalized for a measurement step.
        return {axis: P / np.sqrt(3) for axis, P in
                zip("xyz", (qt.sigmax(), qt.sigmay(), qt.sigmaz()))}


class Spin2AKLTPath(SPTChain):
    bulk_spin = 2

    def __init__(self, t=1.0):
        if not 0 <= t <= 1:
            raise ValueError("t must lie between 0 and 1")
        self.t = t

    def tensors(self):
        # Eq. (16) in the physical Sz basis, followed by Eq. (15).
        r3, r6 = np.sqrt(3), np.sqrt(6)
        matrices = {
            -2: np.diag([r6], -2),
            -1: np.diag([-r3, r3], -1),
             0: np.diag([1, -2, 1]),
             1: np.diag([r3, -r3], 1),
             2: np.diag([r6], 2),
        }
        A = {m: self.t * qt.Qobj(matrix) / np.sqrt(10)
             for m, matrix in matrices.items()}
        A[0] += (1 - self.t) * qt.basis(3, 1).proj()
        return A
