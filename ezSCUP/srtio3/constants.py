"""
Parameters for the SrTiO3 model.
"""

# third party imports
import numpy as np

SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
MASSES = [87.6, 47.9, 16, 16, 16]           # masses, in atomic units 
LABELS = [0, 1, 4, 3, 2]                    # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell
BORN_CHARGES = {                            # Born effective charges
        0: np.array([2.566657, 2.566657, 2.566657]),
        1: np.array([7.265894, 7.265894, 7.265894]),
        4: np.array([-5.707345, -2.062603, -2.062603]),
        3: np.array([-2.062603, -5.707345, -2.062603]),
        2: np.array([-2.062603, -2.062603, -5.707345]),
    }
TIO_DIST = 3.6327940 # Ti-O distance, in bohrs