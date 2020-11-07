"""
Class containing an implementation of the mode-projection algorithm.
"""

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func measure(geom, pattern)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def measure(geom, pattern):

    """
    Projects structural modes (patterns) on a given configuration's
    equilibrium geometry.

    # PATTERN DEFINITION # 

    Patterns are basically a list of weighted atomic displacementes,
    which are projected onto each unit cell within the supercell. This
    projection asigns an amplitude 'a' for the pattern in said unit cell.

    Each atomic is defined as a list of four items:

    [label, hopping, weight, target vector]

    where:

    - label (string): is the label of the atomix species
    - hopping (list): is the cell 'hop' between the current 
    cell and the one where the atom of interest is located.
    - weight (float): is the weight assigned to that motion.
    - target vector (list): vector representing the motion
    of interest, it should be normalized.

    The amplitud for each unit cell is then obtained through:

    a = sum_over_all_displacements(weigth*dot_product(position, target vector))

    Parameters:
    ----------

    - pattern (list): Pattern to be projected.

    Return:
    ----------
        - an array the shape of the supercell with the amplitude 
        of the pattern in each unit cell.

    """

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidMCConfiguration()

    values = np.zeros(geom.supercell)

    for x in range(geom.supercell[0]):
        for y in range(geom.supercell[1]):
            for z in range(geom.supercell[2]):
                cell = np.array([x,y,z])
                for atom in pattern:
                    atom_cell = np.mod(cell + atom[1], geom.supercell)
                    nx, ny, nz = atom_cell
                    values[x,y,z] += atom[2]*np.dot(atom[3], geom.displacements[nx,ny,nz,atom[0],:])
    return values
