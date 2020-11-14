"""
Executes STO simulations for three different biaxial strain restrictions,
temperatures spanning between 20 K and 400 K, in 6x6x6 supercells.
"""

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

TEMPERATURES = np.linspace(20, 300, 10)     # temperatures to simulate
STRAINS = [                                 # strains to simulate
    [+0.03, +0.03, 0.0, 0.0, 0.0, 0.0],
    [+0.00, +0.00, 0.0, 0.0, 0.0, 0.0],
    [-0.03, -0.03, 0.0, 0.0, 0.0, 0.0]
]   # +-3% and 0% cell strain in the x and y direction (Voigt notation)

cfg.MC_STEPS = 1000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 100            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 20                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  
# fixed strain components: xx, yy, xy (Voigt notation)
cfg.FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                     MAIN FUNCTION CALL                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    # create simulation class
    sim = MCSimulation()
    sim.setup(
        "srtio3", "srtio3_full_lat.xml", SUPERCELL, SPECIES, NATS,
        temp = TEMPERATURES, strain=STRAINS,
        output_folder = "output"
    )

    # simulate and properly store output
    sim.sequential_launch_by_temperature(inverse_order=True)

    meas = STOAnalyzer(output_folder = "output")

    print("\nGenerating AFDa plots...", end = '')
    meas.AFD_horizontal_domain_vectors([0,1], mode="a")
    meas.AFD_vs_T(mode="a")
    print(" DONE!")

    print("\nGenerating AFDi plots...", end = '')
    meas.AFD_horizontal_domain_vectors([0,1], mode="i")
    meas.AFD_vs_T(mode="i")
    print(" DONE!")

    print("\nGenerating FE plots...", end = '')
    meas.FE_vs_T(abs=True)
    meas.FE_vs_T()
    print(" DONE!")
    
    print("\nGenerating Polarization plots...", end = '')
    meas.POL_vs_T()
    print(" DONE!")

    print("\nGenerating Strain plots...", end = '')
    meas.STRA_vs_T()
    print(" DONE!")

    print("\nEVERYTHING DONE!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #