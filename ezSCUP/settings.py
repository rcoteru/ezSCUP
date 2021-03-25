"""
General settings for the ezSCUP package.
"""

import os

#####################################################################
##  EZSCUP SETTINGS
#####################################################################

# Location of the SCALE-UP executable in the system
# This setting is required to run any simulations.
# By default picks up the environment variable SCUP_EXEC.
# If it hasnt been setup, defaults to None.
SCUP_EXEC = os.getenv("SCUP_EXEC", default = None) 

# Folder where the SCALE-UP models are located
SCUP_MODELS = os.getenv("SCUP_MODELS", default = None) 

# Whether or not to overwrite old output when starting a 
# new simulation run. 
# default: False
OVERWRITE = False

# simulation info filename, stores parameter vectors of 
# the simulation run in the output folder
SIMULATION_SETUP_FILE = "simulation.info"

# regular expression to use when parsing for lattice data
LT_SEARCH_WORD = "LT:"

#####################################################################
##  MONTE CARLO FDF DEFAULT SETTINGS
#####################################################################

# Total steps for MC simulations
# FDF setting: "mc_nsweeps"
MC_STEPS = 1000

# Equilibration steps for MC simulations
# By default this is set to 0, but running any simulations
# without equilibration steps is highly NOT recommended.
MC_EQUILIBRATION_STEPS = 0

# Step interval for partial .restart file printing in MC simulations.
# FDF setting: "n_write_mc"
MC_STEP_INTERVAL = 20

# Maximum MC jump possible, in Angstrom.
# FDF setting: "mc_max_step_d"
MC_MAX_JUMP = 0.5

# Annealing rate for MC simulations, 
# set to 1 for constant temperature. 
# (expected behaviour)
# FDF setting: "mc_annealing_rate"
MC_ANNEALING_RATE = 1

# Step interval for lattice info in the output.
# FDF setting: "print_std_lattice_nsteps"
LATTICE_OUTPUT_INTERVAL = 50

# Whether or not to fix the array componentes. Requires a list of six 
# boolean values, representing if a component is fixed (True) or 
# not (False) in Voigt notation. 
#
# For example, the following value fixes the xx, yy and xy components:
# FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]
# 
# FDF setting: "fix_strain_component" [block]
# default: [False, False, False, False, False, False]
FIXED_STRAIN_COMPONENTS = [False, False, False, False, False, False]

# Whether or not to print FDF settings before each simulation run. 
PRINT_CONF_SETTINGS = False


#####################################################################
##  GEOMETRY OPTIMIZATION (CGD) FDF DEFAULT SETTINGS
#####################################################################

# TODO DOCUMENTATION

CGD_MAX_ITER = 1000

CGD_FORCE_THRESHOLD = 0.01 # in ev/Ang

CGD_FORCE_FACTOR = 1.0

CGD_STRESS_FACTOR = 10.0

CGD_MAX_STEP = 0.01 # in bohr

