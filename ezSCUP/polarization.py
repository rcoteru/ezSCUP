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
# + func unit_conversion()
# 
# + class BornPolarization()
#   - load()
#   - measure()
#   - restore()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def unit_conversion(scup_polarization):

    e2C = 1.60217646e-19 # elemental charges to Coulombs
    bohr2m = 5.29177e-11 # bohrs to meters

    return scup_polarization*e2C/bohr2m**2 # polarization in C/m2

class BornPolarization():

    """ 
    
    Calculates polarization from a diagonal Born effective charges matrix.

    Attributes:
    ----------

     - config (MCConfiguration):
    
    """

    #######################################################

    def polarization(self, config, born_charges):

        """

        Calculates supercell polarization in the current configuration
        using the given Born effective charges.

        Parameters:
        ----------

        - born_charges (dict): dictionary with element labels as keys
        and effective charge 3D vectors as values. (in elemental charge units)

        Return:
        ----------
            - a 3D vector with the macroscopic polarization (in C/m2)
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        stra = self.config.strains
        ncells = self.config.supercell[0]*self.config.supercell[1]*self.config.supercell[2]

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
        volume = ncells*ucell_volume

        pol = np.zeros(3)
        for x in range(self.config.supercell[0]):
            for y in range(self.config.supercell[1]):
                for z in range(self.config.supercell[2]):

                    for label in self.config.elements:

                        tau = self.config.cells[x,y,z].displacements[label]
                        charges = born_charges[label]
                        
                        for i in range(3):
                            pol[i] += charges[i]*tau[i]

        pol = pol/volume # in e/bohr2
        pol = pol*1.60217646e-19 # to C/bohr
        pol = pol/(5.29177e-11)**2 # to C/m2

        self.pol = pol
        return pol

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def stepped_polarization(self, config, born_charges):

        """

        Calculates supercell polarization in the current configuration
        for every single .partial file using the given Born effective charges.

        Parameters:
        ----------

        - born_charges (dict): dictionary with element labels as keys
        and effective charge 3D vectors as values. (in elemental charge units)

        Return:
        ----------
            - a list of 3D vectors with the macroscopic polarization (in C/m2)

        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        ncells = self.config.supercell[0]*self.config.supercell[1]*self.config.supercell[2]
        resp = RestartParser()

        pol_hist = []
        for f in self.config.partials:
            
            resp.load(f)

            stra = resp.strains

            ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
            volume = ncells*ucell_volume

            pol = np.zeros(3)
            for x in range(self.config.supercell[0]):
                for y in range(self.config.supercell[1]):
                    for z in range(self.config.supercell[2]):

                        for label in self.config.elements:

                            tau = resp.cells[x,y,z].displacements[label]
                            charges = born_charges[label]
                            
                            for i in range(3):
                                pol[i] += charges[i]*tau[i]

            pol = pol/volume # in e/bohr2
            pol = pol*1.60217646e-19 # to C/bohr
            pol = pol/(5.29177e-11)**2 # to C/m2
            pol_hist.append(pol)
        
        self.pol_hist = pol_hist
        return pol_hist

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def layered_polarization(self, config, born_charges):

        """

        Calculates supercell polarization in the current configuration
        in horizontal (z-axis) layers using the given Born effective charges.

        Parameters:
        ----------

        - born_charges (dict): dictionary with element labels as keys
        and effective charge 3D vectors as values. (in elemental charge units)

        Return:
        ----------
            - a list of 3D vectors with the macroscopic polarization (in C/m2)
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        stra = self.config.strains
        ncells_per_layer = self.config.supercell[0]*self.config.supercell[1]

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
        volume = ucell_volume*ncells_per_layer

        pols_by_layer = []
        for layer in range(self.config.supercell[2]):

            pol = np.zeros(3)
            for x in range(self.config.supercell[0]):
                for y in range(self.config.supercell[1]):

                    for label in self.config.elements:

                        tau = self.config.cells[x,y,layer].displacements[label]
                        charges = born_charges[label]
                        
                        for i in range(3):
                            pol[i] += charges[i]*tau[i]

            pol = pol/volume # in e/bohr2
            pol = pol*1.60217646e-19 # to C/bohr
            pol = pol/(5.29177e-11)**2 # to C/m2
            pols_by_layer.append(pol)

        self.pols_by_layer = pols_by_layer
        return pols_by_layer

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def perovs_unit_cell_polarization(self, config, born_charges):

        """

        Calculates supercell polarization in the current configuration
        in a per-unit-cell basis using the given Born effective charges.

        Parameters:
        ----------

        - born_charges (dict): dictionary with element labels as keys
        and effective charge 3D vectors as values. (in elemental charge units)

        Return:
        ----------
            - three supercell-sized arrays containing the polarization in the
            x, y and z direction of each unit cell (in C/m2)

        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        A, B, Ox, Oy, Oz = labels

        FE_mode=[  # atom, hopping, weight
            # "frame"
            [A, [0, 0, 0], 1./8.],
            [A, [1, 0, 0], 1./8.],
            [A, [1, 1, 0], 1./8.],
            [A, [0, 1, 0], 1./8.],
            [A, [0, 0, 1], 1./8.],
            [A, [1, 0, 1], 1./8.],
            [A, [1, 1, 1], 1./8.],
            [A, [0, 1, 1], 1./8.],
            # "octahedra"
            [B, [0, 0, 0], 1.], # b site
            [Ox, [0, 0, 0], 1./2.], 
            [Ox, [1, 0, 0], 1./2.],
            [Oy, [0, 0, 0], 1./2.],
            [Oy, [0, 1, 0], 1./2.],
            [Oz, [0, 0, 0], 1./2.],
            [Oz, [0, 0, 1], 1./2.]
        ]

        cnts = self.config.lat_constants
        stra = self.config.strains

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))

        polx = np.zeros(self.config.supercell)
        poly = np.zeros(self.config.supercell)
        polz = np.zeros(self.config.supercell)

        for x in range(self.config.supercell[0]):
            for y in range(self.config.supercell[1]):
                for z in range(self.config.supercell[2]):

                    pol = np.zeros(3)
                    cell = np.array([x,y,z])

                    for atom in FE_mode:

                        atom_cell = np.mod(cell + atom[1], self.config.supercell)
                        nx, ny, nz = atom_cell

                        tau = self.config.cells[nx,ny,nz].displacements[atom[0]]
                        charges = born_charges[atom[0]]
                        
                        for i in range(3):
                            pol[i] += atom[2]*charges[i]*tau[i]

                    pol = pol/ucell_volume # in e/bohr2
                    pol = pol*1.60217646e-19 # to C/bohr
                    pol = pol/(5.29177e-11)**2 # to C/m2

                    polx[x,y,z] = pol[0]
                    poly[x,y,z] = pol[1]
                    polz[x,y,z] = pol[2]

        
        return polx, poly, polz