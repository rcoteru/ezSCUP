"""

Collection of functions designed to project structural modes, 
such as the antiferrodistortive (AFD) and ferroelectric (FE) modes,
on SrTiO3, There is also a method to obtain the per-unit-cell
polarization.

"""

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.srtio3.constants import LABELS, BORN_CHARGES
from ezSCUP.polarization import unit_conversion
from ezSCUP.projection import measure
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func STO_ROT(config)
# + func STO_AFD(config, mode="a")
# + func STO_FE(config)
# + func STO_AFE(config)
# + func STO_OD(config)
#
# + func perovskite_polarization(config, labels, born_charges)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

A, B, Ox, Oy, Oz = LABELS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

ROT_X=[
        # atom, hopping, weight, target vector
        [Oy, [0, 0, 0],1./4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0],1./4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 0, 1],1./4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 1, 0],1./4.,[ 0.0,-1.0, 0.0]]
    ]

ROT_Y=[
        # atom, hopping, weight, target vector
        [Ox, [0, 0, 0],1./4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 0, 0],1./4.,[-1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0],1./4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 1],1./4.,[ 1.0, 0.0, 0.0]]
    ]

ROT_Z=[
        # atom, hopping, weight, target vector
        [Ox, [0, 0, 0],1./4.,[ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0],1./4.,[ 1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0],1./4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 0],1./4.,[-1.0, 0.0, 0.0]]
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

AFDa_X=[
        # atom, hopping, weight, target vector
        # "lower" cell
        [Oy, [0, 0, 0], 0.7071067812/4., [ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0], 0.7071067812/4., [ 0.0, 1.0, 0.0]],
        [Oy, [0, 0, 1], 0.7071067812/4., [ 0.0, 0.0, 1.0]],
        [Oz, [0, 1, 0], 0.7071067812/4., [ 0.0,-1.0, 0.0]],
        # "upper" cell
        [Oy, [1, 0, 0], 0.7071067812/4., [ 0.0, 0.0, 1.0]],
        [Oz, [1, 0, 0], 0.7071067812/4., [ 0.0,-1.0, 0.0]],
        [Oy, [1, 0, 1], 0.7071067812/4., [ 0.0, 0.0,-1.0]],
        [Oz, [1, 1, 0], 0.7071067812/4., [ 0.0, 1.0, 0.0]]
    ]

AFDa_Y=[
        # atom, hopping, weight, target vector
        # lower cell
        [Ox, [0, 0, 0], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 0, 0], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 1], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        # upper cell
        [Ox, [0, 1, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 1, 0], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        [Ox, [1, 1, 0], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 1, 1], 0.7071067812/4.,[-1.0, 0.0, 0.0]]
    ]

AFDa_Z=[
        # atom, hopping, weight, target vector
        # "lower" cell
        [Ox, [0, 0, 0], 0.7071067812/4.,[ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 0], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        # "upper" cell
        [Ox, [0, 0, 1], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 0, 1], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        [Ox, [1, 0, 1], 0.7071067812/4.,[ 0.0,-1.0, 0.0]],
        [Oy, [0, 1, 1], 0.7071067812/4.,[ 1.0, 0.0, 0.0]]
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

AFDi_X=[
        # atom, hopping, weight, target vector
        # "lower" cell
        [Oy, [0, 0, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 0, 1], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 1, 0], 0.7071067812/4.,[ 0.0,-1.0, 0.0]],
        # "upper" cell
        [Oy, [1, 0, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [1, 0, 0], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [1, 0, 1], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [1, 1, 0], 0.7071067812/4.,[ 0.0,-1.0, 0.0]]
    ]

AFDi_Y=[
        # atom, hopping, weight, target vector
        # lower cell
        [Ox, [0, 0, 0], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 0, 0], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 0, 1], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        # upper cell
        [Ox, [0, 1, 0], 0.7071067812/4.,[ 0.0, 0.0, 1.0]],
        [Oz, [0, 1, 0], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        [Ox, [1, 1, 0], 0.7071067812/4.,[ 0.0, 0.0,-1.0]],
        [Oz, [0, 1, 1], 0.7071067812/4.,[ 1.0, 0.0, 0.0]]
    ]

AFDi_Z=[
        # atom, hopping, weight, target vector
        # "lower" cell
        [Ox, [0, 0, 0], 0.7071067812/4.,[ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        [Ox, [1, 0, 0], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 0], 0.7071067812/4.,[-1.0, 0.0, 0.0]],
        # "upper" cell
        [Ox, [0, 0, 1], 0.7071067812/4.,[ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 1], 0.7071067812/4.,[ 1.0, 0.0, 0.0]],
        [Ox, [1, 0, 1], 0.7071067812/4.,[ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 1], 0.7071067812/4.,[-1.0, 0.0, 0.0]]
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

FE_X=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [1, 0, 0], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [1, 1, 0], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [0, 1, 0], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [0, 0, 1], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [1, 0, 1], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [1, 1, 1], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        [A, [0, 1, 1], 0.4227802165/8., [ 1.0, 0.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1532194652/1., [ 1.0, 0.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.3737995525/2., [-1.0, 0.0, 0.0]], 
        [Ox, [1, 0, 0], 0.3737995525/2., [-1.0, 0.0, 0.0]],
        [Oy, [0, 0, 0], 0.5736090038/2., [-1.0, 0.0, 0.0]],
        [Oy, [0, 1, 0], 0.5736090038/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 0], 0.5736090038/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 1], 0.5736090038/2., [-1.0, 0.0, 0.0]]
    ]

FE_Y=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [1, 0, 0], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [1, 1, 0], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [0, 1, 0], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [0, 0, 1], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [1, 0, 1], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [1, 1, 1], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        [A, [0, 1, 1], 0.4227802165/8., [ 0.0, 1.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1532194652/1., [ 0.0, 1.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.5736090038/2., [ 0.0,-1.0, 0.0]], 
        [Ox, [1, 0, 0], 0.5736090038/2., [ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0], 0.3737995525/2., [ 0.0,-1.0, 0.0]],
        [Oy, [0, 1, 0], 0.3737995525/2., [ 0.0,-1.0, 0.0]],
        [Oz, [0, 0, 0], 0.5736090038/2., [ 0.0,-1.0, 0.0]],
        [Oz, [0, 0, 1], 0.5736090038/2., [ 0.0,-1.0, 0.0]]
    ]

FE_Z=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [1, 0, 0], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [1, 1, 0], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [0, 1, 0], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [0, 0, 1], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [1, 0, 1], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [1, 1, 1], 0.4227802165/8., [0.0, 0.0, 1.0]],
        [A, [0, 1, 1], 0.4227802165/8., [0.0, 0.0, 1.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1532194652/1., [0.0, 0.0, 1.0]], # b site
        [Ox, [0, 0, 0], 0.5736090038/2., [0.0, 0.0,-1.0]], 
        [Ox, [1, 0, 0], 0.5736090038/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 0, 0], 0.5736090038/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 1, 0], 0.5736090038/2., [0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0], 0.3737995525/2., [0.0, 0.0,-1.0]],
        [Oz, [0, 0, 1], 0.3737995525/2., [0.0, 0.0,-1.0]]
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

AFE_X=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [1, 0, 0], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [1, 1, 0], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [0, 1, 0], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [0, 0, 1], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [1, 0, 1], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [1, 1, 1], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        [A, [0, 1, 1], 0.6105737038/8., [-1.0, 0.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.7721715327/1., [ 1.0, 0.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.0073879524/2., [ 1.0, 0.0, 0.0]], 
        [Ox, [1, 0, 0], 0.0073879524/2., [ 1.0, 0.0, 0.0]],
        [Oy, [0, 0, 0], 0.1242905757/2., [-1.0, 0.0, 0.0]],
        [Oy, [0, 1, 0], 0.1242905757/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 0], 0.1242905757/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 1], 0.1242905757/2., [-1.0, 0.0, 0.0]]
    ]

AFE_Y=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 0, 0], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 1, 0], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 1, 0], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 0, 1], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 0, 1], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 1, 1], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 1, 1], 0.6105737038/8., [ 0.0,-1.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.7721715327/1., [ 0.0, 1.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.1242905757/2., [ 0.0,-1.0, 0.0]], 
        [Ox, [1, 0, 0], 0.1242905757/2., [ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0], 0.0073879524/2., [ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 0], 0.0073879524/2., [ 0.0, 1.0, 0.0]],
        [Oz, [0, 0, 0], 0.1242905757/2., [ 0.0,-1.0, 0.0]],
        [Oz, [0, 0, 1], 0.1242905757/2., [ 0.0,-1.0, 0.0]]
    ]

AFE_Z=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [1, 0, 0], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [1, 1, 0], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [0, 1, 0], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [0, 0, 1], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [1, 0, 1], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [1, 1, 1], 0.6105737038/8., [0.0, 0.0,-1.0]],
        [A, [0, 1, 1], 0.6105737038/8., [0.0, 0.0,-1.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.7721715327/1., [0.0, 0.0, 1.0]], # b site
        [Ox, [0, 0, 0], 0.1242905757/2., [0.0, 0.0,-1.0]], 
        [Ox, [1, 0, 0], 0.1242905757/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 0, 0], 0.1242905757/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 1, 0], 0.1242905757/2., [0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0], 0.0073879524/2., [0.0, 0.0, 1.0]],
        [Oz, [0, 0, 1], 0.0073879524/2., [0.0, 0.0, 1.0]]
    ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

OD_X=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [1, 0, 0], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [1, 1, 0], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [0, 1, 0], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [0, 0, 1], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [1, 0, 1], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [1, 1, 1], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        [A, [0, 1, 1], 0.0547352163/8., [-1.0, 0.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1563428017/1., [-1.0, 0.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.8723213685/2., [ 1.0, 0.0, 0.0]], 
        [Ox, [1, 0, 0], 0.8723213685/2., [ 1.0, 0.0, 0.0]],
        [Oy, [0, 0, 0], 0.3252817352/2., [-1.0, 0.0, 0.0]],
        [Oy, [0, 1, 0], 0.3252817352/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 0], 0.3252817352/2., [-1.0, 0.0, 0.0]],
        [Oz, [0, 0, 1], 0.3252817352/2., [-1.0, 0.0, 0.0]]
    ]

OD_Y=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 0, 0], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 1, 0], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 1, 0], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 0, 1], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 0, 1], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [1, 1, 1], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        [A, [0, 1, 1], 0.0547352163/8., [ 0.0,-1.0, 0.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1563428017/1., [ 0.0,-1.0, 0.0]], # b site
        [Ox, [0, 0, 0], 0.3252817352/2., [ 0.0,-1.0, 0.0]], 
        [Ox, [1, 0, 0], 0.3252817352/2., [ 0.0,-1.0, 0.0]],
        [Oy, [0, 0, 0], 0.8723213685/2., [ 0.0, 1.0, 0.0]],
        [Oy, [0, 1, 0], 0.8723213685/2., [ 0.0, 1.0, 0.0]],
        [Oz, [0, 0, 0], 0.3252817352/2., [ 0.0,-1.0, 0.0]],
        [Oz, [0, 0, 1], 0.3252817352/2., [ 0.0,-1.0, 0.0]]
    ]

OD_Z=[  # atom, hopping, weight, target vector
        # "frame"
        [A, [0, 0, 0], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [1, 0, 0], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [1, 1, 0], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [0, 1, 0], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [0, 0, 1], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [1, 0, 1], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [1, 1, 1], 0.0547352163/8., [0.0, 0.0,-1.0]],
        [A, [0, 1, 1], 0.0547352163/8., [0.0, 0.0,-1.0]],
        # "octahedra"
        [B,  [0, 0, 0], 0.1563428017/1., [0.0, 0.0,-1.0]], # b site
        [Ox, [0, 0, 0], 0.3252817352/2., [0.0, 0.0,-1.0]], 
        [Ox, [1, 0, 0], 0.3252817352/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 0, 0], 0.3252817352/2., [0.0, 0.0,-1.0]],
        [Oy, [0, 1, 0], 0.3252817352/2., [0.0, 0.0,-1.0]],
        [Oz, [0, 0, 0], 0.8723213685/2., [0.0, 0.0, 1.0]],
        [Oz, [0, 0, 1], 0.8723213685/2., [0.0, 0.0, 1.0]]
    ]


# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~ Pattern Projection ~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

def STO_ROT(geom, angles=True):

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

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject

    dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
    dist[:,:,:,0] = measure(geom, ROT_X)
    dist[:,:,:,1] = measure(geom, ROT_Y)
    dist[:,:,:,2] = measure(geom, ROT_Z)

    if angles:
        BO_dist = np.linalg.norm(geom.positions[0,0,0,B,:] - geom.positions[0,0,0,Ox,:])
        return np.arctan(dist/BO_dist)*180/np.pi
    else:
        return dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def STO_AFD(geom, mode="a", angles=True):

    """

    Calculates the octahedral antiferrodistortive rotation 
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

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject

    if mode == "a":

        # distortion
        dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
        dist[:,:,:,0] = measure(geom, AFDa_X)
        dist[:,:,:,1] = measure(geom, AFDa_Y)
        dist[:,:,:,2] = measure(geom, AFDa_Z)
        
        # symmetry corrections -> R-point instability
        for x in range(geom.supercell[0]):
            for y in range(geom.supercell[1]):
                for z in range(geom.supercell[2]):
                    dist[x,y,z,:] = (-1)**x * (-1)**y * (-1)**z * dist[x,y,z,:]

        if angles:
            BO_dist = np.linalg.norm(geom.positions[0,0,0,B,:] - geom.positions[0,0,0,Ox,:])
            return np.arctan(dist/(np.sqrt(2)*BO_dist))*180/np.pi
        else:
            return dist

    else:

        # distortion
        dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
        dist[:,:,:,0] = measure(geom, AFDi_X)
        dist[:,:,:,1] = measure(geom, AFDi_Y)
        dist[:,:,:,2] = measure(geom, AFDi_Z)
        
        # symmetry corrections -> M-point instability
        for x in range(geom.supercell[0]):
            for y in range(geom.supercell[1]):
                for z in range(geom.supercell[2]):
                    dist[x,y,z,0] = (-1)**y * (-1)**z * dist[x,y,z,0]
                    dist[x,y,z,1] = (-1)**x * (-1)**z * dist[x,y,z,1]
                    dist[x,y,z,2] = (-1)**x * (-1)**y * dist[x,y,z,2]

        if angles:
            BO_dist = np.linalg.norm(geom.positions[0,0,0,B,:] - geom.positions[0,0,0,Ox,:])
            return np.arctan(dist/(np.sqrt(2)*BO_dist))*180/np.pi
        else:
            return dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



def STO_FE(geom):

    """

    Calculates the ferroelectric displacements of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    
    Return:
    ----------
    - three supercell-sized arrays containing the ferroelectric dispalcements 
    in the x, y and z directions of each unit cell (in bohr)
    
    """

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject

    dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
    dist[:,:,:,0] = measure(geom, FE_X)
    dist[:,:,:,1] = measure(geom, FE_Y)
    dist[:,:,:,2] = measure(geom, FE_Z)

    return dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def STO_AFE(geom):

    """

    Calculates the antiferroelectric displacements of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    
    Return:
    ----------
    - three supercell-sized arrays containing the ferroelectric dispalcements 
    in the x, y and z directions of each unit cell (in bohr)
    
    """

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject

    dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
    dist[:,:,:,0] = measure(geom, AFE_X)
    dist[:,:,:,1] = measure(geom, AFE_Y)
    dist[:,:,:,2] = measure(geom, AFE_Z)

    return dist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def STO_OD(geom):

    """

    Calculates the octahedral distortions of each unit cell.
    
    Parameters:
    ----------

    - config (MCConfiguration): configuration to be loaded.
    - labels (list): identifiers of the five perovskite atoms [A, B, 0x, Oy, Oz]
    
    Return:
    ----------
    - three supercell-sized arrays containing the ferroelectric dispalcements 
    in the x, y and z directions of each unit cell (in bohr)
    
    """

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject
    
    dist = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))
    dist[:,:,:,0] = measure(geom, OD_X)
    dist[:,:,:,1] = measure(geom, OD_Y)
    dist[:,:,:,2] = measure(geom, OD_Z)

    return dist

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~ Polarization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

def STO_POL(geom):

    """

    Calculates supercell polarization in the current configuration
    in a per-cubic-cell basis using the given Born effective charges.

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

    if not isinstance(geom, Geometry):
        raise ezSCUP.exceptions.InvalidGeometryObject

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

    cnts = geom.lat_constants
    stra = geom.strains

    ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))

    pols = np.zeros((geom.supercell[0], geom.supercell[1], geom.supercell[2], 3))

    for x in range(geom.supercell[0]):
        for y in range(geom.supercell[1]):
            for z in range(geom.supercell[2]):

                pol = np.zeros(3)
                cell = np.array([x,y,z])

                for atom in FE_mode:

                    atom_cell = np.mod(cell + atom[1], geom.supercell)
                    nx, ny, nz = atom_cell

                    tau = geom.displacements[nx,ny,nz,atom[0],:]
                    charges = np.array(BORN_CHARGES[atom[0]])
                    
                    for i in range(3):
                        pol[i] += atom[2]*charges[i]*tau[i]

                pol = pol/ucell_volume # in e/bohr2
                pol = unit_conversion(pol)

                pols[x,y,z,:] = pol

    return pols