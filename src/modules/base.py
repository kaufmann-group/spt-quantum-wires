from abc import ABC, abstractmethod

import numpy as np
import qutip as qt


class SPTChain(ABC):
    """Uniform matrix-product chain in a specified physical basis."""

    @abstractmethod
    def tensors(self):
        """Return {physical basis label: virtual bond matrix}."""
        ...

    def transfer_spectrum(self):
        E = sum(qt.sprepost(A, A.dag()) for A in self.tensors().values())
        values = E.eigenenergies()
        return np.sort(np.real_if_close(values / max(abs(values))))
