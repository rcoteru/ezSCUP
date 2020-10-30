"""
Provides a class 
"""

# third party imports
import numpy as np          # matrix support

# standard library imports
import os

# package imports
from ezSCUP.handlers import SP_SCUPHandler, FDFSetting
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func SPRun(parameter_file, geom)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def SPRun(parameter_file, geom):

    name = "SPDefaultName"
    sim = SP_SCUPHandler(name, parameter_file, cfg.SCUP_EXEC)

    sim.settings["supercell"] = [list(geom.supercell)]
    sim.settings["geometry_restart"] =  FDFSetting(name + ".restart")
    geom.write_restart(name + ".restart")
        
    sim.launch(output_file=name + ".out")
        
    f = open(name + ".out")

    line = f.readline().strip()
    while (line != "Energy decomposition:"):
        line = f.readline()
        line = line.strip()

    energy = {} # in eV
    energy["reference"] = float(f.readline().strip().split()[3])
    energy["total_delta"] = float(f.readline().strip().split()[3])

    energy["lat_total_delta"] = float(f.readline().strip().split()[3])
    energy["lat_harmonic"] = float(f.readline().strip().split()[2])
    energy["lat_anharmonic"] = float(f.readline().strip().split()[2])
    energy["lat_elastic"] = float(f.readline().strip().split()[2])
    energy["lat_electrostatic"] = float(f.readline().strip().split()[2])

    energy["elec_total_delta"] = float(f.readline().strip().split()[3])
    energy["elec_one_electron"] = float(f.readline().strip().split()[2])
    energy["elec_two_electron"] = float(f.readline().strip().split()[2])
    energy["elec_electron_lat"] = float(f.readline().strip().split()[2])
    energy["elec_electrostatic"] = float(f.readline().strip().split()[2])

    energy["total_energy"] = float(f.readline().strip().split()[3])
    f.close()

    # cleanup
    os.remove(name + ".restart")
    os.remove(name + ".out")
    os.remove(name + "_FINAL.REF")
    os.remove(name + "_FINAL.restart")
        
    return energy