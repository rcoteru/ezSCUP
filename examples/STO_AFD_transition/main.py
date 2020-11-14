"""
Executes STO simulations in a temperature range from 20 to 400K in order
to observe its AFD phase transition.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~ REQUIRED MODULE IMPORTS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# standard library imports
import os, sys
import time

# third party imports
import numpy as np

# ezSCUP imports
from ezSCUP.srtio3.constants import SPECIES, NATS
from ezSCUP.srtio3.analysis import STOAnalyzer
from ezSCUP.montecarlo import MCSimulation
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       SIMULATION SETTINGS                         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: Location of the SCALE-UP executable in the system
SCUP_EXEC = os.getenv("SCUP_EXEC", default = None) 
OVERWRITE = False                           # overwrite old output folder?

SUPERCELL = [4,4,4]                         # shape of the supercell

TEMPERATURES = np.linspace(20, 400, 20)

cfg.MC_STEPS = 1000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 100            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 20                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  
cfg.MC_MAX_JUMP = 0.15                      # MC max jump size (in Angstrom)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                        MAIN FUNCTION CALL                         #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# create simulation class
sim = MCSimulation()
sim.setup(
    "srtio3", "srtio3_full_lat.xml", SUPERCELL, SPECIES, NATS,
    temp = TEMPERATURES, output_folder = "output"
)
    
# simulate from high to low temp
sim.sequential_launch_by_temperature(inverse_order=True)

meas = STOAnalyzer(output_folder = "output")

print("\nGenerating AFDa plots...")
meas.AFD_vs_T(mode="a", rotate=True)

print("\nGenerating AFDi plots...")
meas.AFD_vs_T(mode="i", rotate=True)

print("\nGenerating FE plots...")
meas.FE_vs_T()

print("\nGenerating Polarization plots...")
meas.POL_vs_T()

print("\nGenerating Strain plots...")
meas.STRA_vs_T()

print("\nEVERYTHING DONE!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #