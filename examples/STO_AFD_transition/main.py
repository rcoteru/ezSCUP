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

if __name__ == "__main__":

    # create simulation class
    sim = MCSimulation()
    sim.setup(
        "srtio3", "srtio3_full_lat.xml", SUPERCELL, SPECIES, NATS,
        temp = TEMPERATURES, output_folder = "output"
    )
        
    # simulate from high to low temp
    sim.sequential_launch_by_temperature(inverse_order=True)

    meas = PKAnalyzer(LABELS, BORN_CHARGES, output_folder = "output")

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