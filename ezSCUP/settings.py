"""
General settings for the ezSCUP package.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

#####################################################################
##  EZSCUP SETTINGS
#####################################################################

# Location of the Scale-Up executable in the system
# This setting is required to run any simulations.
# default: None
SCUP_EXEC = None

# Equilibration steps for MC simulations
# By default this disabled, but running any simulations
# without equilibration steps is highly NOT recommended.
# default: None
MC_EQUILIBRATION_STEPS = None

# simulation info filename, stores parameter vectors of 
# the simulation run in the output folder
SIMULATION_SETUP_FILE = "simulation.info"

# regular expression to use when parsing for lattice data
LT_SEARCH_WORD = "LT:"

# whether to print individual configuration settings, mainly 
# for debugging purposes
PRINT_CONF_SETTINGS = True

#####################################################################
##  FDF SETTINGS
#####################################################################

# Total steps for MC simulations
# If None, reads it from the FDF file.
# FDF setting: "mc_nsweeps"
# default: None
MC_STEPS = None

# Step interval for partial .restart file printing in MC simulations.
# If None, reads it from the FDF file.
# FDF setting: "n_write_mc"
# default: None
MC_STEP_INTERVAL = None

# Maximum MC jump possible, in Angstrom,
# If None, reads it from the FDF file.
# FDF setting: "mc_max_step_d"
# default: None
MC_MAX_JUMP = None

# Step interval for lattice info in the output.
# If None, reads it from the FDF file.
# FDF setting: "print_std_lattice_nsteps"
# default: None 
LATTICE_OUTPUT_INTERVAL = None

# Whether or not to fix the array componentes. Requires a list of six 
# boolean values, representing if a component is fixed (True) or 
# not (False) in Voigt notation. 
#
# For example, the following value fixes the xx, yy and xy components:
# FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]
# 
# If None, reads it from the FDF file.
# FDF setting: "fix_strain_component" [block]
# default: None
FIXED_STRAIN_COMPONENTS = None
