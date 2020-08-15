"""

Test script for the mode-projection algortihm:

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
from ezSCUP.generators import RestartGenerator
from ezSCUP.perovskite import perovskite_AFD
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up-1.0.0/build_dir/src/scaleup.x"

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [4,4,4]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
LABELS = ["Sr", "Ti", "O3", "O2", "O1"]     # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell

TEMPERATURES = np.linspace(20, 100, 5)      # temperatures to simulate

cfg.MC_STEPS = 4000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 500            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 20                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  

# no fixed strain components
cfg.FIXED_STRAIN_COMPONENTS = [False]*6

#cfg.OVERWRITE = True                        # whether to overwrite previous output folders
#cfg.PRINT_CONF_SETTINGS = True              # (DEBUGGING) print individual FDF settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    sim = MCSimulation()
    sim.setup(
        "input.fdf", SUPERCELL, SPECIES, NATS,
        temp = TEMPERATURES, output_folder = "output"
    )

    # test sequential simulation run (reverse order)
    sim.sequential_launch_by_temperature(inverse_order=True)

    
    parser = MCSimulationParser(output_folder = "output")

    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "AFD_x", "AFD_y", "AFD_z"))
    
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        AFDa_x, AFDa_y, AFDa_z = perovskite_AFD(config, LABELS)
        AFDa_x = np.mean(AFDa_x)
        AFDa_y = np.mean(AFDa_y)
        AFDa_z = np.mean(AFDa_z)


        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, AFDa_x, AFDa_y, AFDa_z))


    print("\nEVERYTHING DONE!")
