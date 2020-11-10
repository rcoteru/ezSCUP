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
from ezSCUP.perovskite.analysis import PKAnalyzer
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
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
LABELS = [0, 1, 4, 3, 2]                    # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell
BORN_CHARGES = {                            # Born effective charges
        0: np.array([2.566657, 2.566657, 2.566657]),
        1: np.array([7.265894, 7.265894, 7.265894]),
        4: np.array([-5.707345, -2.062603, -2.062603]),
        3: np.array([-2.062603, -5.707345, -2.062603]),
        2: np.array([-2.062603, -2.062603, -5.707345]),
    }

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

    meas = PKAnalyzer(LABELS, BORN_CHARGES, output_folder = "output")

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