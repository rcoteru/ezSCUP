"""
Collection of classes for general analysis of output data.
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

# func octahedra_rotation()

#####################################################################
## MODE ANALYZER
#####################################################################

class ModeAnalyzer():

    """ 
    
    Projects structural modes (patterns) on a given configuration's
    equilibrium geometry.

    # BASIC USAGE # 

    

    # PATTERN DEFINITION #

    Patterns are basically a list of weighted atomic displacementes,


    [label, hopping, weight, target vector]

    Attributes:
    ----------

     - supercell (array)
     - config (MCConfiguration):

     - new_sc (array): 
     - unit_cell (array):
     - variable (array): 
    
    """

    def load(self, config):

        """


        
        Parameters:
        ----------

        - config (MCConfiguration): 
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        self.supercell = config.supercell

    #######################################################

    def measure(self, new_sc, unit_cell, variable):

        """


        
        Parameters:
        ----------

        - new_sc (array): 
        - unit_cell (array):
        - variable (array): 
        
        """

        self.new_sc = new_sc
        self.unit_cell = unit_cell
        self.variable = variable

        values = np.zeros(new_sc)

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):

                    cell = np.array([x,y,z])

                    for atom in self.variable:

                        atom_cell = np.mod(cell + atom[1], self.supercell)

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
    
    Calculates polarization from diagonal Born Charges.

    # BASIC USAGE # 

    # PATTERN DEFINITION #

    [label, hopping, weight, target vector]

    Attributes:
    ----------

     - supercell (array)
     - config (MCConfiguration):

     - new_sc (array): 
     - unit_cell (array):
     - variable (array): 
    
    """

    def load(self, config):

        """


        
        Parameters:
        ----------

        - config (MCConfiguration): 
        
        """

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        self.supercell = config.supercell

    #######################################################

    def polarization(self, born_charges):

        """



        Parameters:
        ----------

        - born_charges (array):
        
        """

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        stra = self.config.strains
        ncells = self.supercell[0]*self.supercell[1]*self.supercell[2]

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
        volume = ncells*ucell_volume

        pol = np.zeros(3)
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):

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



        Parameters:
        ----------

        - born_charges (array):
        
        """

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        ncells = self.supercell[0]*self.supercell[1]*self.supercell[2]
        resp = RestartParser()

        pol_hist = []
        for f in self.config.partials:
            
            resp.load(f)

            stra = resp.strains

            ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
            volume = ncells*ucell_volume

            pol = np.zeros(3)
            for x in range(self.supercell[0]):
                for y in range(self.supercell[1]):
                    for z in range(self.supercell[2]):

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



        Parameters:
        ----------

        - born_charges (array):
        
        """

        labels = list(born_charges.keys())
    
        for l in labels:
            if l not in self.config.elements:
                raise ezSCUP.exceptions.InvalidLabel

        
        cnts = self.config.lat_constants
        stra = self.config.strains
        ncells_per_layer = self.supercell[0]*self.supercell[1]

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
        volume = ucell_volume*ncells_per_layer

        pols_by_layer = []
        for layer in range(self.supercell[2]):

            pol = np.zeros(3)
            for x in range(self.supercell[0]):
                for y in range(self.supercell[1]):

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



        Parameters:
        ----------

        - born_charges (array):
        
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

        polx = np.zeros(self.supercell)
        poly = np.zeros(self.supercell)
        polz = np.zeros(self.supercell)

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):

                    pol = np.zeros(3)
                    cell = np.array([x,y,z])

                    for atom in FE_mode:

                        atom_cell = np.mod(cell + atom[1], self.supercell)
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
    
    Parameters:
    ----------

    
    
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

    unit_cell=[[1,0,0],[0,1,0],[0,0,1]]

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

        x_distortions = analyzer.measure(config.supercell, unit_cell, AFDa_X)
        x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

        y_distortions = analyzer.measure(config.supercell, unit_cell, AFDa_Y)
        y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

        z_distortions = analyzer.measure(config.supercell, unit_cell, AFDa_Z)
        z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

        for x in range(config.supercell[0]):
            for y in range(config.supercell[1]):
                for z in range(config.supercell[2]):
                    x_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

    else:

        x_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_X)
        x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

        y_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_Y)
        y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

        z_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_Z)
        z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

        for x in range(config.supercell[0]):
            for y in range(config.supercell[1]):
                for z in range(config.supercell[2]):
                    x_angles[x,y,z] = (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

def perovskite_FE_full(config, labels, mode="B"):

    #TODO DOCUMENTATION

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

    unit_cell=[[1,0,0],[0,1,0],[0,0,1]]

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


    x_dist = analyzer.measure(config.supercell, unit_cell, FE_X_B)
    y_dist = analyzer.measure(config.supercell, unit_cell, FE_Y_B)
    z_dist = analyzer.measure(config.supercell, unit_cell, FE_Z_B)

    return x_dist, y_dist, z_dist


def perovskite_FE_simple(config, labels, mode="B"):

    #TODO DOCUMENTATION

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

    unit_cell=[[1,0,0],[0,1,0],[0,0,1]]

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

        x_dist = analyzer.measure(config.supercell, unit_cell, FE_X_A)
        y_dist = analyzer.measure(config.supercell, unit_cell, FE_Y_A)
        z_dist = analyzer.measure(config.supercell, unit_cell, FE_Z_A)

        return x_dist, y_dist, z_dist

    else:

        x_dist = analyzer.measure(config.supercell, unit_cell, FE_X_B)
        y_dist = analyzer.measure(config.supercell, unit_cell, FE_Y_B)
        z_dist = analyzer.measure(config.supercell, unit_cell, FE_Z_B)

        return x_dist, y_dist, z_dist


def perovskite_simple_rotation(config, labels):

    #TODO DOCUMENTATION

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

    unit_cell=[[1,0,0],[0,1,0],[0,0,1]]

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

    x_distortions = analyzer.measure(config.supercell, unit_cell, rot_X)
    x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

    y_distortions = analyzer.measure(config.supercell, unit_cell, rot_Y)
    y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

    z_distortions = analyzer.measure(config.supercell, unit_cell, rot_Z)
    z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

    return (x_angles, y_angles, z_angles)


