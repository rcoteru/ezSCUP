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
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func afdaxy_vortex(supercell, species, labels, disp, region_size)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~ Geometry Generation ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

def afda_xy_vortex(supercell, species, labels, disp, region_size):

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


    geom = Geometry(supercell, species, 5)

    _, B, Ox, Oy, Oz = labels 

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
                    geom.displacements[x,y,z,B,0] -= disp
                    geom.displacements[x,y,z,B,1] += disp

                if case == "B":
                    # rotación x negativa
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    # rotación y
                    geom.displacements[x,y,z,Oz,1] += factor*disp
                    geom.displacements[x,y,z,Oy,2] -= factor*disp
                    # FE x negativo y negativo
                    geom.displacements[x,y,z,B,0] -= disp
                    geom.displacements[x,y,z,B,1] -= disp

                if case == "D":
                    # rotación x negativa
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    geom.displacements[x,y,z,Oy,2] += factor*disp
                    # FE x positivo y negativo
                    geom.displacements[x,y,z,B,0] += disp
                    geom.displacements[x,y,z,B,1] -= disp

                if case == "C":
                    # rotación x
                    geom.displacements[x,y,z,Ox,2] -= factor*disp
                    geom.displacements[x,y,z,Oz,0] += factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    geom.displacements[x,y,z,Oy,2] += factor*disp 
                    # FE x positivo y positivo
                    geom.displacements[x,y,z,B,0] += disp
                    geom.displacements[x,y,z,B,1] += disp

    return geom