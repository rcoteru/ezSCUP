"""
    
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
# + func AFDaxy_vortex(supercell, species, labels, disp, region_size)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class STOGeometry(Geometry):

    def __init__(self, supercell, model):

        super().__init__(supercell, model["species"], model["nats"])

        self.lat_vectors = np.zeros((3,3))
        self.lat_constants = np.zeros(3)
        for i in range(3):
            self.lat_vectors[i,:] = np.array(model["lat_vectors"][i])
            self.lat_constants[i] = np.linalg.norm(self.lat_vectors[i,:])

        sc = self.supercell
        self.positions = np.zeros([sc[0], sc[1], sc[2], self.nats, 3])
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    cell_vector = x*self.lat_vectors[0,:] + y*self.lat_vectors[1,:] + z*self.lat_vectors[2,:]
                    for j in range(self.nats):
                        self.positions[x,y,z,j,:] = cell_vector + model["ref_struct"][j]

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~ GEOMETRY GENERATION ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

def STO_AFD(supercell, model, angle, mode="a", axis="z", clockwise=False):

    """

    """

    _, _, Ox, Oy, Oz = model["labels"]
    geom = STOGeometry(supercell, model)
    disp = model["BOdist"]*np.arctan(angle/180.*np.pi)

    if clockwise:
        cw = 1
    else:
        cw = 0

    for x in range(supercell[0]):
        for y in range(supercell[1]):
            for z in range(supercell[2]):

                if mode == "a":

                    factor = (-1)**x * (-1)**y * (-1)**z * (-1)**cw

                    if axis == "x":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                    elif axis == "y":
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                    elif axis == "z":
                        geom.displacements[x,y,z,Ox,1] -= factor*disp
                        geom.displacements[x,y,z,Oy,0] += factor*disp
                    elif axis == "xy" or axis == "yx":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                    else:
                        raise NotImplementedError()
                elif mode == "i":

                    if axis == "x":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                    elif axis == "y":
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                    elif axis == "z":
                        factor = (-1)**x * (-1)**y * (-1)**cw
                        geom.displacements[x,y,z,Ox,1] -= factor*disp
                        geom.displacements[x,y,z,Oy,0] += factor*disp
                    elif axis == "xy" or axis == "yx":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                    else:
                        raise NotImplementedError()
    
                else:
                    raise NotImplementedError()

    return geom

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def STO_FE(supercell, model, disp, axis="z"):

    """

    """

    _, B, _, _, _ = model["labels"]
    geom = STOGeometry(supercell, model)

    for x in range(supercell[0]):
        for y in range(supercell[1]):
            for z in range(supercell[2]):

                if axis == "x":
                    geom.displacements[x,y,z,B,0] += disp
                elif axis == "y":
                    geom.displacements[x,y,z,B,1] += disp
                elif axis == "z":
                    geom.displacements[x,y,z,B,2] += disp
                elif axis == "xy" or axis == "yx":
                    geom.displacements[x,y,z,B,0] += disp
                    geom.displacements[x,y,z,B,1] += disp
                else:
                    raise NotImplementedError()

    return geom

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def STO_AFD_FE(supercell, model, angle, ti_disp, mode="a", axis="z", clockwise=False):

    """

    """

    _, B, Ox, Oy, Oz = model["labels"]
    geom = STOGeometry(supercell, model)
    disp = model["BOdist"]*np.arctan(angle/180.*np.pi)

    if clockwise:
        cw = 1
    else:
        cw = 0

    for x in range(supercell[0]):
        for y in range(supercell[1]):
            for z in range(supercell[2]):

                if mode == "a":

                    factor = (-1)**x * (-1)**y * (-1)**z * (-1)**cw

                    if axis == "x":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        geom.displacements[x,y,z,B,0]  += ti_disp
                    elif axis == "y":
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    elif axis == "z":
                        geom.displacements[x,y,z,Ox,1] -= factor*disp
                        geom.displacements[x,y,z,Oy,0] += factor*disp
                        geom.displacements[x,y,z,B,2]  += ti_disp
                    elif axis == "xy" or axis == "yx":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                        geom.displacements[x,y,z,B,0]  += ti_disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    else:
                        raise NotImplementedError()

                elif mode == "i":

                    if axis == "x":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        geom.displacements[x,y,z,B,0]  += ti_disp
                    elif axis == "y":
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    elif axis == "z":
                        factor = (-1)**x * (-1)**y * (-1)**cw
                        geom.displacements[x,y,z,Ox,1] -= factor*disp[1]
                        geom.displacements[x,y,z,Oy,0] += factor*disp[0]
                        geom.displacements[x,y,z,B,2]  += ti_disp
                    elif axis == "xy" or axis == "yx":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                        geom.displacements[x,y,z,B,0]  += ti_disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    else:
                        raise NotImplementedError()
    
                else:
                    raise NotImplementedError()

    return geom


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def AFD_FE_xy_vortex(supercell, model, angle, region_size, Ti_disp=0.3):

    """
    Generates a rotational AFDa vortex structure in the xy plane.

    This geometry consists of alternating regions of compound xy axis
    rotations of the oxygen octahedra, resulting in the creation of vortexes
    while projecting the AFDa rotational mode. The structure also generates
    a polarization in the xy plain, which is also included for stability 
    purposes.

    Parameters:
    ----------

    - supercell (array): supercell shape
    - species (list): atomic species within the supercell
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    - disp (real): displacement of the oxygen atoms in the rotation, in bohrs.
    - region_size (int)

    Return:
    ----------
    - the requested Geometry object.
    """

    _, B, Ox, Oy, Oz = model["labels"]
    geom = STOGeometry(supercell, model)
    disp = model["BOdist"]*np.arctan(angle/180.*np.pi)

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
                    geom.displacements[x,y,z,Ox,2] -= factor*disp
                    geom.displacements[x,y,z,Oz,0] += factor*disp
                    # rotación y
                    geom.displacements[x,y,z,Oz,1] += factor*disp
                    geom.displacements[x,y,z,Oy,2] -= factor*disp
                    # FE x negativo y positivo
                    geom.displacements[x,y,z,B,0] += Ti_disp
                    geom.displacements[x,y,z,B,1] -= Ti_disp

                if case == "B":
                    # rotación x negativa
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    # rotación y
                    geom.displacements[x,y,z,Oz,1] += factor*disp
                    geom.displacements[x,y,z,Oy,2] -= factor*disp
                    # FE x negativo y negativo
                    geom.displacements[x,y,z,B,0] += Ti_disp
                    geom.displacements[x,y,z,B,1] += Ti_disp

                if case == "D":
                    # rotación x negativa
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    geom.displacements[x,y,z,Oy,2] += factor*disp
                    # FE x positivo y negativo
                    geom.displacements[x,y,z,B,0] -= Ti_disp
                    geom.displacements[x,y,z,B,1] += Ti_disp

                if case == "C":
                    # rotación x
                    geom.displacements[x,y,z,Ox,2] -= factor*disp
                    geom.displacements[x,y,z,Oz,0] += factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    geom.displacements[x,y,z,Oy,2] += factor*disp 
                    # FE x positivo y positivo
                    geom.displacements[x,y,z,B,0] -= Ti_disp
                    geom.displacements[x,y,z,B,1] -= Ti_disp

    return geom


