"""
Class containing an implementation of the mode-projection algorithm.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.simulations import MCConfiguration
from ezSCUP.generators import RestartGenerator
from ezSCUP.geometry import Geometry, UnitCell

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class ModeAnalyzer()
#   - measure(config, pattern)
#   - restore() # TODO
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ModeAnalyzer():

    """ 
    
    Projects structural modes (patterns) on a given configuration's
    equilibrium geometry.

    # BASIC USAGE # 

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

    Attributes:
    ----------
    
     - config (MCConfiguration): loaded configuration.

     - variable (array): last projected pattern.
    
    """


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def measure(self, config, pattern):

        """

        Projects the introduced pattern onto a previously loaded supercell.
        
        Parameters:
        ----------

        - pattern (list): Pattern to be projected.

        Return:
        ----------
            - an array the shape of the supercell with the amplitude 
            of the pattern in each unit cell.
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration()

        self.config = config
        self.pattern = pattern

        values = np.zeros(self.config.geo.supercell)

        for x in range(self.config.geo.supercell[0]):
            for y in range(self.config.geo.supercell[1]):
                for z in range(self.config.geo.supercell[2]):

                    cell = np.array([x,y,z])

                    for atom in self.pattern:

                        atom_cell = np.mod(cell + atom[1], self.config.geo.supercell)

                        nx, ny, nz = atom_cell

                        values[x,y,z] += atom[2]*np.dot(atom[3], 
                            self.config.geo.cells[nx,ny,nz].displacements[atom[0]])

        self.values = values
        return values

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def restore(self, config, pattern, values=None):

        """
        
        Reconstructs a geometry from values and pattern
        
        """

        # TODO think this one out

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration()

        if values == None:
            values = self.measure(config, pattern)

        RestartGenerator(config.supercell, config.species, config.nats)

        for x in range(self.config.supercell[0]):
            for y in range(self.config.supercell[1]):
                for z in range(self.config.supercell[2]):
                    pass

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

