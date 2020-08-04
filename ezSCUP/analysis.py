"""
Collection of classes for general analysis of output data.

Contains two classes to both project structural modes and 
calculate macroscopic polarizations from born effective charges.
There are also functions to analyze the AFD and FE modes of perovskites.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.simulations import MCConfiguration
from ezSCUP.structures import FDFSetting, Cell
from ezSCUP.parsers import RestartParser
import ezSCUP.settings as cfg
import ezSCUP.exceptions

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class ModeAnalyzer()
# class BornPolarization()

# func perovskite_AFD()
# func perovskite_FE_full()
# func perovskite_FE_simple()
# func perovskite_simple_rotation()

#####################################################################
## MODE ANALYZER
#####################################################################

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

    def load(self, config):

        """

        Loads a configuration for its later use.
        
        Parameters:
        ----------

        - config (MCConfiguration): configuration to be loaded.
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

    #######################################################

    def measure(self, pattern):

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

        self.pattern = pattern

        values = np.zeros(self.config.supercell)

        for x in range(self.config.supercell[0]):
            for y in range(self.config.supercell[1]):
                for z in range(self.config.supercell[2]):

                    cell = np.array([x,y,z])

                    for atom in self.pattern:

                        atom_cell = np.mod(cell + atom[1], self.config.supercell)

                        nx, ny, nz = atom_cell

                        values[x,y,z] += atom[2]*np.dot(atom[3], self.config.cells[nx,ny,nz].displacements[atom[0]])

        self.values = values
        return values

        #######################################################

#####################################################################
## POLARIZATION FROM BORN CHARGES
#####################################################################


class BornPolarization():

    """ 
    
    Calculates polarization from a diagonal Born effective charges matrix.

    Attributes:
    ----------

     - config (MCConfiguration):
    
    """

    def load(self, config):

        """

        Loads a configuration for its later use.
        
        Parameters:
        ----------

        - config (MCConfiguration): configuration to be loaded.
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

    #######################################################

    def polarization(self, born_charges):

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

    def stepped_polarization(self, born_charges):

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

    def layered_polarization(self, born_charges):

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

        
    def perovs_unit_cell_polarization(self, born_charges):

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


#####################################################################
## FUNCTION DEFINITIONS
#####################################################################

def perovskite_AFD(config, labels, mode="a"):

    """

    Calculates the perovskite octahedral antiferrodistortive rotation 
    of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    - mode ("a" or "i"): mode of interest, either in-phase or anti-phase (default).
    
    Return:
        ----------
            - three supercell-sized arrays containing the rotation angle in the
            x, y and z directions of each unit cell (in degrees)
    
    """

    if mode != "a" and mode != "i":
        raise NotImplementedError

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    if not isinstance(labels, list):
        raise ezSCUP.exceptions.InvalidLabelList
    
    for l in labels:
        if l not in config.elements:
            raise ezSCUP.exceptions.InvalidLabel

    B, Ox, Oy, Oz = labels[1:]

    analyzer = ModeAnalyzer()
    analyzer.load(config)    

    cell_zero = config.cells[0,0,0]
    BO_dist = np.linalg.norm(cell_zero.positions[B] - cell_zero.positions[Oz])

    AFDa_X=[
            # atom, hopping, weight, target vector
            # "lower" cell
            [Oy, [0, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 0, 1],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 1, 0],1./8.,[ 0.0,-1.0, 0.0]],
            # "upper" cell
            [Oy, [1, 0, 0],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [1, 0, 0],1./8.,[ 0.0,-1.0, 0.0]],
            [Oy, [1, 0, 1],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [1, 1, 0],1./8.,[ 0.0, 1.0, 0.0]]
        ]

    AFDa_Y=[
            # atom, hopping, weight, target vector
            # lower cell
            [Ox, [0, 0, 0],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 0, 0],1./8.,[-1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 1],1./8.,[ 1.0, 0.0, 0.0]],
            # upper cell
            [Ox, [0, 1, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 1, 0],1./8.,[ 1.0, 0.0, 0.0]],
            [Ox, [1, 1, 0],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 1, 1],1./8.,[-1.0, 0.0, 0.0]]
        ]

    AFDa_Z=[
            # atom, hopping, weight, target vector
            # "lower" cell
            [Ox, [0, 0, 0],1./8.,[ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 0],1./8.,[ 1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 1, 0],1./8.,[-1.0, 0.0, 0.0]],
            # "upper" cell
            [Ox, [0, 0, 1],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 0, 1],1./8.,[-1.0, 0.0, 0.0]],
            [Ox, [1, 0, 1],1./8.,[ 0.0,-1.0, 0.0]],
            [Oy, [0, 1, 1],1./8.,[ 1.0, 0.0, 0.0]]
        ]

    AFDi_X=[
            # atom, hopping, weight, target vector
            # "lower" cell
            [Oy, [0, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 0, 1],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 1, 0],1./8.,[ 0.0,-1.0, 0.0]],
            # "upper" cell
            [Oy, [1, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [1, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [1, 0, 1],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [1, 1, 0],1./8.,[ 0.0,-1.0, 0.0]]
        ]

    AFDi_Y=[
            # atom, hopping, weight, target vector
            # lower cell
            [Ox, [0, 0, 0],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 0, 0],1./8.,[-1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 1],1./8.,[ 1.0, 0.0, 0.0]],
            # upper cell
            [Ox, [0, 1, 0],1./8.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 1, 0],1./8.,[-1.0, 0.0, 0.0]],
            [Ox, [1, 1, 0],1./8.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 1, 1],1./8.,[ 1.0, 0.0, 0.0]]
        ]

    AFDi_Z=[
            # atom, hopping, weight, target vector
            # "lower" cell
            [Ox, [0, 0, 0],1./8.,[ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 0],1./8.,[ 1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 1, 0],1./8.,[-1.0, 0.0, 0.0]],
            # "upper" cell
            [Ox, [0, 0, 1],1./8.,[ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 1],1./8.,[ 1.0, 0.0, 0.0]],
            [Ox, [1, 0, 1],1./8.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 1, 1],1./8.,[-1.0, 0.0, 0.0]]
        ]

    if mode == "a":

        x_distortions = analyzer.measure(AFDa_X)
        x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

        y_distortions = analyzer.measure(AFDa_Y)
        y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

        z_distortions = analyzer.measure(AFDa_Z)
        z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

        for x in range(config.supercell[0]):
            for y in range(config.supercell[1]):
                for z in range(config.supercell[2]):
                    x_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

    else:

        x_distortions = analyzer.measure(AFDi_X)
        x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

        y_distortions = analyzer.measure(AFDi_Y)
        y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

        z_distortions = analyzer.measure(AFDi_Z)
        z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

        for x in range(config.supercell[0]):
            for y in range(config.supercell[1]):
                for z in range(config.supercell[2]):
                    x_angles[x,y,z] = (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

def perovskite_FE_full(config, labels):

    """

    Calculates the perovskite ferroelectric displacements of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    
    Return:
        ----------
            - three supercell-sized arrays containing the ferroelectric dispalcements 
            in the x, y and z directions of each unit cell (in bohr)
    
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    if not isinstance(labels, list):
        raise ezSCUP.exceptions.InvalidLabelList

    if len(labels) != 5:
        raise ezSCUP.exceptions.InvalidLabelList
    
    for l in labels:
        if l not in config.elements:
            raise ezSCUP.exceptions.InvalidLabel

    A, B, Ox, Oy, Oz = labels

    analyzer = ModeAnalyzer()
    analyzer.load(config)    

    ### B SITE DISTORTIONS ###

    FE_X_B=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [1, 0, 0], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [1, 1, 0], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [0, 1, 0], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [0, 0, 1], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [1, 0, 1], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [1, 1, 1], 1./8., [ 1.0, 0.0, 0.0]],
            [A, [0, 1, 1], 1./8., [ 1.0, 0.0, 0.0]],
            # "octahedra"
            [B, [0, 0, 0], 1., [ 1.0, 0.0, 0.0]], # b site
            [Ox, [0, 0, 0], 1./2., [-1.0, 0.0, 0.0]], 
            [Ox, [1, 0, 0], 1./2., [-1.0, 0.0, 0.0]],
            [Oy, [0, 0, 0], 1./2., [-1.0, 0.0, 0.0]],
            [Oy, [0, 1, 0], 1./2., [-1.0, 0.0, 0.0]],
            [Oz, [0, 0, 0], 1./2., [-1.0, 0.0, 0.0]],
            [Oz, [0, 0, 1], 1./2., [-1.0, 0.0, 0.0]]
        ]

    FE_Y_B=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [1, 0, 0], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [1, 1, 0], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [0, 1, 0], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [0, 0, 1], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [1, 0, 1], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [1, 1, 1], 1./8., [ 0.0, 1.0, 0.0]],
            [A, [0, 1, 1], 1./8., [ 0.0, 1.0, 0.0]],
            # "octahedra"
            [B, [0, 0, 0], 1., [ 0.0, 1.0, 0.0]], # b site
            [Ox, [0, 0, 0], 1./2., [ 0.0,-1.0, 0.0]], 
            [Ox, [1, 0, 0], 1./2., [ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 0], 1./2., [ 0.0,-1.0, 0.0]],
            [Oy, [0, 1, 0], 1./2., [ 0.0,-1.0, 0.0]],
            [Oz, [0, 0, 0], 1./2., [ 0.0,-1.0, 0.0]],
            [Oz, [0, 0, 1], 1./2., [ 0.0,-1.0, 0.0]]
        ]

    FE_Z_B=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./8., [0.0, 0.0, 1.0]],
            [A, [1, 0, 0], 1./8., [0.0, 0.0, 1.0]],
            [A, [1, 1, 0], 1./8., [0.0, 0.0, 1.0]],
            [A, [0, 1, 0], 1./8., [0.0, 0.0, 1.0]],
            [A, [0, 0, 1], 1./8., [0.0, 0.0, 1.0]],
            [A, [1, 0, 1], 1./8., [0.0, 0.0, 1.0]],
            [A, [1, 1, 1], 1./8., [0.0, 0.0, 1.0]],
            [A, [0, 1, 1], 1./8., [0.0, 0.0, 1.0]],
            # "octahedra"
            [B, [0, 0, 0], 1., [0.0, 0.0, 1.0]], # b site
            [Ox, [0, 0, 0], 1./2., [0.0, 0.0,-1.0]], 
            [Ox, [1, 0, 0], 1./2., [0.0, 0.0,-1.0]],
            [Oy, [0, 0, 0], 1./2., [0.0, 0.0,-1.0]],
            [Oy, [0, 1, 0], 1./2., [0.0, 0.0,-1.0]],
            [Oz, [0, 0, 0], 1./2., [0.0, 0.0,-1.0]],
            [Oz, [0, 0, 1], 1./2., [0.0, 0.0,-1.0]]
        ]


    x_dist = analyzer.measure(FE_X_B)
    y_dist = analyzer.measure(FE_Y_B)
    z_dist = analyzer.measure(FE_Z_B)

    return x_dist, y_dist, z_dist


def perovskite_FE_simple(config, labels, mode="B"):

    """

    Calculates the simple perovskite ferroelectric displacements of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    - mode ("A" or "B"): look at either A-site or B-site (default) displacements.
    
    Return:
        ----------
            - three supercell-sized arrays containing the ferroelectric dispalcements 
            in the x, y and z directions of each unit cell (in bohr)
    
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    if not isinstance(labels, list):
        raise ezSCUP.exceptions.InvalidLabelList

    if len(labels) != 5:
        raise ezSCUP.exceptions.InvalidLabelList
    
    for l in labels:
        if l not in config.elements:
            raise ezSCUP.exceptions.InvalidLabel

    A, B, Ox, Oy, Oz = labels

    analyzer = ModeAnalyzer()
    analyzer.load(config)    

    ### A SITE DISTORTIONS ###

    FE_X_A=[  # atom, hopping, weight, target vector
            [A, [0, 0, 0], 1., [ 1.0, 0.0, 0.0]] # b site
        ]

    FE_Y_A=[  # atom, hopping, weight, target vector
            [A, [0, 0, 0], 1., [ 0.0, 1.0, 0.0]] # b site
        ]

    FE_Z_A=[  # atom, hopping, weight, target vector
            [A, [0, 0, 0], 1., [0.0, 0.0, 1.0]] # b site
        ]

    ### B SITE DISTORTIONS ###

    FE_X_B=[  # atom, hopping, weight, target vector
            [B, [0, 0, 0], 1, [ 1.0, 0.0, 0.0]] # b site
        ]

    FE_Y_B=[  # atom, hopping, weight, target vector
            [B, [0, 0, 0], 1., [ 0.0, 1.0, 0.0]] # b site
        ]

    FE_Z_B=[  # atom, hopping, weight, target vector
            [B, [0, 0, 0], 1., [0.0, 0.0, 1.0]] # b site
        ]

    if mode == "A":

        x_dist = analyzer.measure(FE_X_A)
        y_dist = analyzer.measure(FE_Y_A)
        z_dist = analyzer.measure(FE_Z_A)

        return x_dist, y_dist, z_dist

    else:

        x_dist = analyzer.measure(FE_X_B)
        y_dist = analyzer.measure(FE_Y_B)
        z_dist = analyzer.measure(FE_Z_B)

        return x_dist, y_dist, z_dist


def perovskite_simple_rotation(config, labels):

    """
    Calculates the perovskite octahedral rotations of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    
    Return:
        ----------
            - three supercell-sized arrays containing the rotation angle in the
            x, y and z directions of each unit cell (in degrees)
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    if not isinstance(labels, list):
        raise ezSCUP.exceptions.InvalidLabelList

    if len(labels) != 5:
        raise ezSCUP.exceptions.InvalidLabelList
    
    for l in labels:
        if l not in config.elements:
            raise ezSCUP.exceptions.InvalidLabel

    B, Ox, Oy, Oz = labels[1:]

    analyzer = ModeAnalyzer()
    analyzer.load(config)    

    cell_zero = config.cells[0,0,0]
    BO_dist = np.linalg.norm(cell_zero.positions[B] - cell_zero.positions[Oz])

    ### DISTORTIONS ###

    rot_X=[
            # atom, hopping, weight, target vector
            [Oy, [0, 0, 0],1./4.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 0],1./4.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 0, 1],1./4.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 1, 0],1./4.,[ 0.0,-1.0, 0.0]]
        ]

    rot_Y=[
            # atom, hopping, weight, target vector
            [Ox, [0, 0, 0],1./4.,[ 0.0, 0.0, 1.0]],
            [Oz, [0, 0, 0],1./4.,[-1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./4.,[ 0.0, 0.0,-1.0]],
            [Oz, [0, 0, 1],1./4.,[ 1.0, 0.0, 0.0]]
        ]

    rot_Z=[
            # atom, hopping, weight, target vector
            [Ox, [0, 0, 0],1./4.,[ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 0],1./4.,[ 1.0, 0.0, 0.0]],
            [Ox, [1, 0, 0],1./4.,[ 0.0, 1.0, 0.0]],
            [Oy, [0, 1, 0],1./4.,[-1.0, 0.0, 0.0]]
    ]

    x_distortions = analyzer.measure(rot_X)
    x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

    y_distortions = analyzer.measure(rot_Y)
    y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

    z_distortions = analyzer.measure(rot_Z)
    z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

    return (x_angles, y_angles, z_angles)


