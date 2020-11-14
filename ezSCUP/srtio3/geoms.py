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
from ezSCUP.srtio3.constants import SPECIES, NATS, LABELS, TIO_DIST
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

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~ Geometry Generation ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

def AFD_monodomain(supercell, angle, mode="a", axis="z", clockwise=False):

    """

    """

    disp = TIO_DIST*np.arctan(angle/180.*np.pi)

    geom = Geometry(supercell, SPECIES, NATS)
    _, _, Ox, Oy, Oz = LABELS

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


def AFD_FE_xy_vortex(supercell, angle, region_size, Ti_disp=0.3):

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

    disp = TIO_DIST*np.arctan(angle/180.*np.pi)

    geom = Geometry(supercell, SPECIES, 5)

    _, B, Ox, Oy, Oz = LABELS

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
                    #geom.displacements[x,y,z,B,0] -= Ti_disp
                    #geom.displacements[x,y,z,B,1] += Ti_disp
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
                    #geom.displacements[x,y,z,B,0] -= Ti_disp
                    #geom.displacements[x,y,z,B,1] -= Ti_disp
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
                    #geom.displacements[x,y,z,B,0] += Ti_disp
                    #geom.displacements[x,y,z,B,1] -= Ti_disp
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
                    #geom.displacements[x,y,z,B,0] += Ti_disp
                    #geom.displacements[x,y,z,B,1] += Ti_disp
                    geom.displacements[x,y,z,B,0] -= Ti_disp
                    geom.displacements[x,y,z,B,1] -= Ti_disp

    return geom