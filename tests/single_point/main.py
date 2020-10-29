"""

Test script for the mode-projection and polarization algorithms:

PROJECTIONS:
- simple rotation
- AFDa/AFDi rotation
- full-FE/simple-FE displacement

POLARIZATION:
- supercell-wide polarization
- layered polarization
- stepped polarization
- unit-cell polarization

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

# ezSCUP imports
from ezSCUP.singlepoint import SPSimulation
from ezSCUP.perovskite import generate_vortex_geo

from ezSCUP.polarization import polarization, stepped_polarization, layered_polarization
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up-1.0.0/build_dir/src/scaleup.x"

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [4,4,2]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
LABELS = ["Sr", "Ti", "O3", "O2", "O1"]     # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

sim = SPSimulation()
sim.setup("test", "srtio3_full_lat.xml", SUPERCELL)

angle = 4
BOdist = 3.632794
disp = BOdist*np.tan(angle*np.pi/180.)

gen = generate_vortex_geo(SUPERCELL, SPECIES, LABELS, disp, 2)
sim.run(gen)

os.remove("SP-test.out")
os.remove("test_FINAL.REF")
os.remove("test_FINAL.restart")

print(sim.energy)
