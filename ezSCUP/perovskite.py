"""

Collection of functions designed to project structural modes, 
such as the antiferrodistortive (AFD) and ferroelectric (FE) modes,
on AB03 perovskites. There is also a method to obtain the per-unit-cell
polarization.

"""

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.montecarlo import MCConfiguration
from ezSCUP.generators import RestartGenerator
from ezSCUP.polarization import unit_conversion
from ezSCUP.projection import ModeAnalyzer

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func perovskite_AFD(config, labels, mode="a")
# + func perovskite_FE_full(config, labels)
# + func perovskite_FE_simple(config, labels, mode="B")
# + func perovskite_simple_rotation(config, labels)
# + func perovskite_polarization(config, labels, born_charges)
#
# + generate_vortex_geo()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    B, Ox, Oy, Oz = labels[1:]

    analyzer = ModeAnalyzer()

    cell = config.geo.cells[0,0,0] # just a random cell
    BO_dist_x = np.linalg.norm(cell.positions[B] - cell.positions[Ox])*(1+config.geo.strains[0])
    BO_dist_y = np.linalg.norm(cell.positions[B] - cell.positions[Oy])*(1+config.geo.strains[1])
    BO_dist_z = np.linalg.norm(cell.positions[B] - cell.positions[Oz])*(1+config.geo.strains[2])

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

        x_distortions = analyzer.measure(config, AFDa_X)
        x_angles = np.arctan(x_distortions/BO_dist_x)*180/np.pi

        y_distortions = analyzer.measure(config, AFDa_Y)
        y_angles = np.arctan(y_distortions/BO_dist_y)*180/np.pi

        z_distortions = analyzer.measure(config, AFDa_Z)
        z_angles = np.arctan(z_distortions/BO_dist_z)*180/np.pi

        for x in range(config.geo.supercell[0]):
            for y in range(config.geo.supercell[1]):
                for z in range(config.geo.supercell[2]):
                    x_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * (-1)**z * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

    else:

        x_distortions = analyzer.measure(config, AFDi_X)
        x_angles = np.arctan(x_distortions/BO_dist_x)*180/np.pi

        y_distortions = analyzer.measure(config, AFDi_Y)
        y_angles = np.arctan(y_distortions/BO_dist_y)*180/np.pi

        z_distortions = analyzer.measure(config, AFDi_Z)
        z_angles = np.arctan(z_distortions/BO_dist_z)*180/np.pi

        for x in range(config.geo.supercell[0]):
            for y in range(config.geo.supercell[1]):
                for z in range(config.geo.supercell[2]):
                    x_angles[x,y,z] = (-1)**y * (-1)**z * x_angles[x,y,z]
                    y_angles[x,y,z] = (-1)**x * (-1)**z * y_angles[x,y,z]
                    z_angles[x,y,z] = (-1)**x * (-1)**y * z_angles[x,y,z]

        return (x_angles, y_angles, z_angles)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    A, B, Ox, Oy, Oz = labels

    analyzer = ModeAnalyzer()

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


    x_dist = analyzer.measure(config, FE_X_B)
    y_dist = analyzer.measure(config, FE_Y_B)
    z_dist = analyzer.measure(config, FE_Z_B)

    return x_dist, y_dist, z_dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    A, B, _, _, _ = labels

    analyzer = ModeAnalyzer()

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

        x_dist = analyzer.measure(config, FE_X_A)
        y_dist = analyzer.measure(config, FE_Y_A)
        z_dist = analyzer.measure(config, FE_Z_A)

        return x_dist, y_dist, z_dist

    else:

        x_dist = analyzer.measure(config, FE_X_B)
        y_dist = analyzer.measure(config, FE_Y_B)
        z_dist = analyzer.measure(config, FE_Z_B)

        return x_dist, y_dist, z_dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    B, Ox, Oy, Oz = labels[1:]

    analyzer = ModeAnalyzer()

    cell = config.geo.cells[0,0,0] # just a random cell
    BO_dist_x = np.linalg.norm(cell.positions[B] - cell.positions[Ox])*(1+config.geo.strains[0])
    BO_dist_y = np.linalg.norm(cell.positions[B] - cell.positions[Oy])*(1+config.geo.strains[1])
    BO_dist_z = np.linalg.norm(cell.positions[B] - cell.positions[Oz])*(1+config.geo.strains[2])

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

    x_distortions = analyzer.measure(config, rot_X)
    x_angles = np.arctan(x_distortions/BO_dist_x)*180/np.pi

    y_distortions = analyzer.measure(config, rot_Y)
    y_angles = np.arctan(y_distortions/BO_dist_y)*180/np.pi

    z_distortions = analyzer.measure(config, rot_Z)
    z_angles = np.arctan(z_distortions/BO_dist_z)*180/np.pi

    return (x_angles, y_angles, z_angles)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def perovskite_polarization(config, labels, born_charges):

    """

    Calculates supercell polarization in the current configuration
    in a per-unit-cell basis using the given Born effective charges.

    Parameters:
    ----------

    - born_charges (dict): dictionary with element labels as keys
    and effective charge 3D vectors as values. (in elemental charge units)
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]

    Return:
    ----------
        - three supercell-sized arrays containing the polarization in the
        x, y and z direction of each unit cell (in C/m2)

    
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    config = config

    for l in labels:
        if l not in config.geo.elements:
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

    cnts = config.geo.lat_constants
    stra = config.geo.strains

    ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))

    polx = np.zeros(config.geo.supercell)
    poly = np.zeros(config.geo.supercell)
    polz = np.zeros(config.geo.supercell)

    for x in range(config.geo.supercell[0]):
        for y in range(config.geo.supercell[1]):
            for z in range(config.geo.supercell[2]):

                pol = np.zeros(3)
                cell = np.array([x,y,z])

                for atom in FE_mode:

                    atom_cell = np.mod(cell + atom[1], config.geo.supercell)
                    nx, ny, nz = atom_cell

                    tau = config.geo.cells[nx,ny,nz].displacements[atom[0]]
                    charges = born_charges[atom[0]]
                    
                    for i in range(3):
                        pol[i] += atom[2]*charges[i]*tau[i]

                pol = pol/ucell_volume # in e/bohr2
                pol = unit_conversion(pol)

                polx[x,y,z] = pol[0]
                poly[x,y,z] = pol[1]
                polz[x,y,z] = pol[2]

    
    return polx, poly, polz


def generate_vortex_geo(supercell, species, labels, disp, region_size):


    """

    Generate a rotational AFDa vortex structure in the z direction.

    This geometry consists of alternating regions of compound xy axis
    rotations of the oxygen octahedra, resulting in the creation of vortexes
    while projecting the AFDa rotational mode.

    Parameters:
    ----------

    - supercell (array): supercell shape
    - species (list): atomic species within the supercell
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    - disp (real): displacement of the oxygen atoms in the rotation, in bohrs.
    - region_size (int)

    Return:
    ----------
    - a geometry generator (RestartGenerator) of the requested geometry.

    """


    gen = RestartGenerator(supercell, species, 5)

    _, _, Ox, Oy, Oz = labels 

    for x in range(supercell[0]):
        for y in range(supercell[1]):
            for z in range(supercell[2]):

                factor = (-1)**x * (-1)**y * (-1)**z

                if np.floor(x/region_size)%2 == 0:
                    if np.floor(y/region_size)%2 == 0:
                        case = "A"
                    else:
                        case = "B"
                else:
                    if np.floor(y/region_size)%2 == 0:
                        case = "C"
                    else:
                        case = "D"

                if case == "A":
                    # rotación x
                    gen.cells[x,y,z].displacements[Ox][2] -= factor*disp  
                    gen.cells[x,y,z].displacements[Oz][0] += factor*disp
                    # rotación y
                    gen.cells[x,y,z].displacements[Oz][1] += factor*disp 
                    gen.cells[x,y,z].displacements[Oy][2] -= factor*disp 

                if case == "B":
                    # rotación x negativa
                    gen.cells[x,y,z].displacements[Ox][2] += factor*disp  
                    gen.cells[x,y,z].displacements[Oz][0] -= factor*disp
                    # rotación y
                    gen.cells[x,y,z].displacements[Oz][1] += factor*disp 
                    gen.cells[x,y,z].displacements[Oy][2] -= factor*disp 

                if case == "D":
                    # rotación x negativa
                    gen.cells[x,y,z].displacements[Ox][2] += factor*disp  
                    gen.cells[x,y,z].displacements[Oz][0] -= factor*disp
                    # rotación y negativa
                    gen.cells[x,y,z].displacements[Oz][1] -= factor*disp 
                    gen.cells[x,y,z].displacements[Oy][2] += factor*disp 

                if case == "C":
                    # rotación x
                    gen.cells[x,y,z].displacements[Ox][2] -= factor*disp  
                    gen.cells[x,y,z].displacements[Oz][0] += factor*disp
                    # rotación y negativa
                    gen.cells[x,y,z].displacements[Oz][1] -= factor*disp 
                    gen.cells[x,y,z].displacements[Oy][2] += factor*disp 

    return gen