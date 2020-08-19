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
from ezSCUP.simulations import MCSimulation, MCSimulationParser

from ezSCUP.perovskite import perovskite_simple_rotation, perovskite_AFD
from ezSCUP.perovskite import perovskite_FE_full, perovskite_FE_simple
from ezSCUP.perovskite import perovskite_polarization

from ezSCUP.polarization import polarization, stepped_polarization, layered_polarization
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
BORN_CHARGES = {                            # Born effective charges
        "Sr": np.array([2.566657, 2.566657, 2.566657]),
        "Ti": np.array([7.265894, 7.265894, 7.265894]),
        "O3": np.array([-5.707345, -2.062603, -2.062603]),
        "O2": np.array([-2.062603, -5.707345, -2.062603]),
        "O1": np.array([-2.062603, -2.062603, -5.707345]),
    }

TEMPERATURES = np.linspace(20, 100, 5)      # temperatures to simulate

cfg.MC_STEPS = 4000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 500            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 20                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  

# no fixed strain components
cfg.FIXED_STRAIN_COMPONENTS = [False]*6

CHECK_MODES = True
CHECK_POLARIZATION = True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def mode_checks():

    # simple rotation
    print("\n#### SIMPLE ROTATION TEST (absolute value) ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "AFDa_x", "AFDa_y", "AFDa_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        AFDa_x, AFDa_y, AFDa_z = perovskite_simple_rotation(config, LABELS)
        AFDa_x = np.mean(np.abs(AFDa_x))
        AFDa_y = np.mean(np.abs(AFDa_y))
        AFDa_z = np.mean(np.abs(AFDa_z))

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, AFDa_x, AFDa_y, AFDa_z))

    # AFDa mode
    print("\n#### AFDa MODE TEST ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "AFDa_x", "AFDa_y", "AFDa_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        AFDa_x, AFDa_y, AFDa_z = perovskite_AFD(config, LABELS)
        AFDa_x = np.mean(AFDa_x)
        AFDa_y = np.mean(AFDa_y)
        AFDa_z = np.mean(AFDa_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, AFDa_x, AFDa_y, AFDa_z))

    # AFDi mode
    print("\n#### AFDi MODE TEST ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "AFDi_x", "AFDi_y", "AFDi_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        AFDi_x, AFDi_y, AFDi_z = perovskite_AFD(config, LABELS, mode="i")
        AFDi_x = np.mean(AFDi_x)
        AFDi_y = np.mean(AFDi_y)
        AFDi_z = np.mean(AFDi_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, AFDi_x, AFDi_y, AFDi_z))
    

    # full-FE mode
    print("\n#### FULL-FE MODE TEST ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "FE_x", "FE_y", "FE_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        FE_x, FE_y, FE_z = perovskite_FE_full(config, LABELS)
        FE_x = np.mean(FE_x)
        FE_y = np.mean(FE_y)
        FE_z = np.mean(FE_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, FE_x, FE_y, FE_z))

    # simple-FE mode (B-site)
    print("\n#### SIMPLE-FE MODE (B-SITE) TEST ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "FE_x", "FE_y", "FE_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        FE_x, FE_y, FE_z = perovskite_FE_simple(config, LABELS, mode="B")
        FE_x = np.mean(FE_x)
        FE_y = np.mean(FE_y)
        FE_z = np.mean(FE_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, FE_x, FE_y, FE_z))

    # simple-FE mode (A-site)
    print("\n#### SIMPLE-FE MODE (A-SITE) TEST ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "FE_x", "FE_y", "FE_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        FE_x, FE_y, FE_z = perovskite_FE_simple(config, LABELS, mode="A")
        FE_x = np.mean(FE_x)
        FE_y = np.mean(FE_y)
        FE_z = np.mean(FE_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, FE_x, FE_y, FE_z))
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def polarization_checks():

    # supercell-wide polarization
    print("\n#### SUPERCELL POLARIZATION ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "pol_x", "pol_y", "pol_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)

        pol_x, pol_y, pol_z = polarization(config, BORN_CHARGES)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, pol_x, pol_y, pol_z))

    # stepped polarization polarization
    print("\n#### PARTIAL-WISE POLARIZATION ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "pol_x", "pol_y", "pol_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        pol_x = 0
        pol_y = 0
        pol_z = 0

        pol_hist = stepped_polarization(config, BORN_CHARGES)
        for pol in pol_hist:
            pol_x += pol[0]
            pol_y += pol[1]
            pol_z += pol[2]

        pol_x = pol_x / len(pol_hist)
        pol_y = pol_y / len(pol_hist)
        pol_z = pol_z / len(pol_hist)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, pol_x, pol_y, pol_z))


    # stepped polarization polarization
    print("\n#### LAYERED POLARIZATION ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "pol_x", "pol_y", "pol_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        pol_x = 0
        pol_y = 0
        pol_z = 0

        pol_hist = layered_polarization(config, BORN_CHARGES)
        for pol in pol_hist:
            pol_x += pol[0]
            pol_y += pol[1]
            pol_z += pol[2]

        pol_x = pol_x / len(pol_hist)
        pol_y = pol_y / len(pol_hist)
        pol_z = pol_z / len(pol_hist)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, pol_x, pol_y, pol_z))

    # stepped polarization polarization
    print("\n#### UNIT-CELL POLARIZATION ####")
    print("\n{:>15}{:>15}{:>15}{:>15}".format("Temperature", "pol_x", "pol_y", "pol_z"))
    for t in TEMPERATURES:
        
        config = parser.access(t=t)
        
        pol_x, pol_y, pol_z = perovskite_polarization(config, LABELS, BORN_CHARGES)
        pol_x = np.mean(pol_x)
        pol_y = np.mean(pol_y)
        pol_z = np.mean(pol_z)

        print("{:15f}{:15.5f}{:15.5f}{:15.5f}".format(t, pol_x, pol_y, pol_z))
    

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

if __name__ == "__main__":

    sim = MCSimulation()
    sim.setup(
        "input.fdf", SUPERCELL, SPECIES, NATS,
        temp = TEMPERATURES, output_folder = "output"
    )

    # sequential simulation run (reverse order)
    sim.sequential_launch_by_temperature(inverse_order=True)

    parser = MCSimulationParser(output_folder = "output")

    if CHECK_MODES:
        mode_checks()

    if CHECK_POLARIZATION:
        polarization_checks()

    print("\nEVERYTHING DONE!")
