"""

Test script for the basic funtionality of the package.

- program and execute several simulation runs
- check output folder swap procedure
- check independent simulation run procedure
- check sequential simulation run procedure (direct and reversed)
- check application of starting geometry

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
from ezSCUP.montecarlo import MCSimulation, MCSimulationParser
from ezSCUP.geometry import Geometry
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up-1.0.0/build_dir/src/scaleup.x"

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [2,2,2]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
LABELS = [0, 1, 4, 3, 2]                    # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell

TEMPERATURES = np.linspace(20, 100, 2)      # temperatures to simulate
STRESSES = [                                # stresses to simulate
    [10., 10., 10., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.]
]   # 10 GPa strains in every direction
STRAINS = [                                 # strains to simulate
    [+0.03, +0.03, 0.0, 0.0, 0.0, 0.0],
    [-0.03, -0.03, 0.0, 0.0, 0.0, 0.0]
]   # +-3% x and y cell strain
FIELDS = [                                  # electric fields to simulate
    [0., 0., 1e9],
    [0., 0., 0.]
]   # 1e9 V/m = 1V/nm in the z direction  

cfg.MC_STEPS = 100                          # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 11             # MC equilibration steps
cfg.MC_STEP_INTERVAL = 10                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  

# fixed strain components: xx, yy, xy (Voigt notation)
cfg.FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]

cfg.OVERWRITE = True                         # whether to overwrite previous output folders
#cfg.PRINT_CONF_SETTINGS = True              # (DEBUGGING) print individual FDF settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

sim = MCSimulation()
sim.setup(
    "srtio3", "srtio3_full_lat.xml",  
    SUPERCELL, SPECIES, NATS,
    temp = TEMPERATURES, strain = STRAINS,
    stress= STRESSES, field= FIELDS,
    output_folder = "output"
)

A, B, Ox, Oy, Oz = LABELS

# create an example starting geometry
start_geom = Geometry(SUPERCELL, SPECIES, NATS)

for x in range(SUPERCELL[0]):
    for y in range(SUPERCELL[1]):
        for z in range(SUPERCELL[2]):
            start_geom.displacements[x,y,z,B,:] = [0., 0., 0.2]

# test independent simulation run
sim.change_output_folder("independent_output")
sim.independent_launch(start_geo=start_geom)

# test sequential simulation run
sim.change_output_folder("sequential_output")
sim.sequential_launch_by_temperature(start_geo=start_geom)

# test sequential simulation run (reverse order)
sim.change_output_folder("sequential_output_reverse")
sim.sequential_launch_by_temperature(
    start_geo=start_geom,
    inverse_order=True)

print("\nEVERYTHING DONE!")
