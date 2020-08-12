"""

Test script for an independent simulation run.

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
from ezSCUP.simulations import MCSimulation, MCSimulationParser
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up-1.0.0/build_dir/src/scaleup.x"

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [2,2,2]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                # elements in the lattice
NATS = 5                                    # number of atoms per cell

TEMPERATURES = np.linspace(20, 60, 3)     # temperatures to simulate
STRAINS = [                                 # strains to simulate
    [+0.03, +0.03, 0.0, 0.0, 0.0, 0.0],
    [+0.00, +0.00, 0.0, 0.0, 0.0, 0.0],
    [-0.03, -0.03, 0.0, 0.0, 0.0, 0.0]
]   # +-3% and 0% cell strain in the x and y direction (Voigt notation)

cfg.MC_STEPS = 100                          # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 10             # MC equilibration steps
cfg.MC_STEP_INTERVAL = 10                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  

# fixed strain components: xx, yy, xy (Voigt notation)
cfg.FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]

cfg.OVERWRITE = True                         # whther to overwrite previous output folders
#cfg.PRINT_CONF_SETTINGS = True              # (DEBUGGING) print individual FDF settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    sim = MCSimulation()
    sim.setup(
        "input.fdf", SUPERCELL, SPECIES, NATS,
        temp = TEMPERATURES, strain = STRAINS,
        output_folder = "output"
    )

    sim.change_output_folder("independent_output")
    sim.independent_launch()

    sim.change_output_folder("sequential_output")
    sim.sequential_launch_by_temperature()

    print("\nEVERYTHING DONE!")