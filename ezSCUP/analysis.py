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
import ezSCUP.settings as cfg
import ezSCUP.exceptions

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class ModeProjector()

# func octahedra_rotation()

#####################################################################
## MODE ANALYZER
#####################################################################

class ModeAnalyzer():

    """ TODO DOCUMENTATION """

    config = None    
    
    supercell = []

    new_supercell = []
    variable = []
    unit_cell = []

    def load(self, config):

        """TODO DOCUMENTATION"""

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        self.supercell = config.supercell

    #######################################################

    def measure(self, new_sc, unit_cell, variable):

        """TODO DOCUMENTATION"""

        self.new_sc = new_sc
        self.unit_cell = unit_cell
        self.variable = variable

        values = np.zeros(new_sc)

        for x in range(new_sc[0]):
            for y in range(new_sc[1]):
                for z in range(new_sc[2]):

                    cell = np.dot([x,y,z], self.supercell)

                    for atom in self.variable:

                        atom_cell = np.mod(cell + atom[1], self.supercell)

                        nx, ny, nz = atom_cell

                        values[x,y,z] += atom[2]*np.dot(atom[3],
                            self.config.cells[nx,ny,nz].displacements[atom[0]])

        self.values = values
        return values

        #######################################################


#####################################################################
## FUNCTION DEFINITIONS
#####################################################################

def perovskite_AFD(config, labels, mode="a"):

    #TODO DOCUMENTATION

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

        return (x_angles, y_angles, z_angles)

    else:

        x_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_X)
        x_angles = np.arctan(x_distortions/BO_dist)*180/np.pi

        y_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_Y)
        y_angles = np.arctan(y_distortions/BO_dist)*180/np.pi

        z_distortions = analyzer.measure(config.supercell, unit_cell, AFDi_Z)
        z_angles = np.arctan(z_distortions/BO_dist)*180/np.pi

        return (x_angles, y_angles, z_angles)

def perovskite_FE(config, labels):

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

    ### DISTORTIONS ###

    FE_X=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [1, 0, 0], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [1, 1, 0], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [0, 1, 0], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [0, 0, 1], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [1, 0, 1], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [1, 1, 1], 1./15., [ 1.0, 0.0, 0.0]],
            [A, [0, 1, 1], 1./15., [ 1.0, 0.0, 0.0]],
            # "octahedra"
            [B, [0, 0, 0], 1./15., [ 1.0, 0.0, 0.0]],
            [Ox, [0, 0, 0], 1./15., [-1.0, 0.0, 0.0]], 
            [Ox, [1, 0, 0], 1./15., [-1.0, 0.0, 0.0]],
            [Oy, [0, 0, 0], 1./15., [-1.0, 0.0, 0.0]],
            [Oy, [0, 1, 0], 1./15., [-1.0, 0.0, 0.0]],
            [Oz, [0, 0, 0], 1./15., [-1.0, 0.0, 0.0]],
            [Oz, [0, 0, 1], 1./15., [-1.0, 0.0, 0.0]]
        ]

    FE_Y=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [1, 0, 0], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [1, 1, 0], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [0, 1, 0], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [0, 0, 1], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [1, 0, 1], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [1, 1, 1], 1./15., [ 0.0, 1.0, 0.0]],
            [A, [0, 1, 1], 1./15., [ 0.0, 1.0, 0.0]],
            # "octahedra"
            [B, [0, 0, 0], 1./15., [ 0.0, 1.0, 0.0]],
            [Ox, [0, 0, 0], 1./15., [ 0.0,-1.0, 0.0]], 
            [Ox, [1, 0, 0], 1./15., [ 0.0,-1.0, 0.0]],
            [Oy, [0, 0, 0], 1./15., [ 0.0,-1.0, 0.0]],
            [Oy, [0, 1, 0], 1./15., [ 0.0,-1.0, 0.0]],
            [Oz, [0, 0, 0], 1./15., [ 0.0,-1.0, 0.0]],
            [Oz, [0, 0, 1], 1./15., [ 0.0,-1.0, 0.0]]
        ]

    FE_Z=[  # atom, hopping, weight, target vector
            # "frame"
            [A, [0, 0, 0], 1./15., [0.0, 0.0, 1.0]],
            [A, [1, 0, 0], 1./15., [0.0, 0.0, 1.0]],
            [A, [1, 1, 0], 1./15., [0.0, 0.0, 1.0]],
            [A, [0, 1, 0], 1./15., [0.0, 0.0, 1.0]],
            [A, [0, 0, 1], 1./15., [0.0, 0.0, 1.0]],
            [A, [1, 0, 1], 1./15., [0.0, 0.0, 1.0]],
            [A, [1, 1, 1], 1./15., [0.0, 0.0, 1.0]],
            [A, [0, 1, 1], 1./15., [0.0, 0.0, 1.0]],
            # "octahedra"
            [B, [0, 0, 0], 1./15., [0.0, 0.0, 1.0]],
            [Ox, [0, 0, 0], 1./15., [0.0, 0.0,-1.0]], 
            [Ox, [1, 0, 0], 1./15., [0.0, 0.0,-1.0]],
            [Oy, [0, 0, 0], 1./15., [0.0, 0.0,-1.0]],
            [Oy, [0, 1, 0], 1./15., [0.0, 0.0,-1.0]],
            [Oz, [0, 0, 0], 1./15., [0.0, 0.0,-1.0]],
            [Oz, [0, 0, 1], 1./15., [0.0, 0.0,-1.0]]
        ]

    x_dist = analyzer.measure(config.supercell, unit_cell, FE_X)
    y_dist = analyzer.measure(config.supercell, unit_cell, FE_Y)
    z_dist = analyzer.measure(config.supercell, unit_cell, FE_Z)

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

