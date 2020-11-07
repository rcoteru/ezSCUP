"""

Test script for the mode-projection and polarization algorithms:

"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~ REQUIRED MODULE IMPORTS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# standard library imports
import os, sys
import time

# third party imports
import matplotlib.pyplot as plt
import numpy as np
import pickle

# ezSCUP imports
from ezSCUP.singlepoint import SPRun
from ezSCUP.geometry import Geometry
from ezSCUP.normodes import finite_hessian
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up-1.0.0/build_dir/src/scaleup.x"

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [1,1,1]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
MASSES = [87.6, 47.9, 16, 16, 16]           # masses, in atomic units 
LABELS = [0, 1, 4, 3, 2]                    # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

run = False
geo = Geometry(SUPERCELL, SPECIES, NATS)

if run == True:
    hessian = finite_hessian("srtio3_full_lat.xml", geo, MASSES)
    with open("hessian.pickle", "wb") as f:
                pickle.dump(hessian, f)
else:
    with open("hessian.pickle", "rb") as f:
                hessian = pickle.load(f) 

M = []
for m in MASSES:
    for _ in range(3):    
        M.append(m)
M = np.diagflat(M)
Mi = np.linalg.inv(M)

normHessian = np.matmul(Mi, hessian)
w, v = np.linalg.eigh(normHessian)

np.set_printoptions(suppress=True, precision=3)
for i, val in enumerate(w):
    print("---------------------------------------------")
    print("Mode #" + str(i) + " -> Eigenvalue:" + str(val))
    print("Sr: " + str(v[0:3,i]))
    print("Ti: " + str(v[3:6,i]))
    print("Oz: " + str(v[6:9,i]))
    print("Oy: " + str(v[9:12,i]))
    print("Ox: " + str(v[12:15,i]))
    print("---------------------------------------------\n")