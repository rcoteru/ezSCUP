"""
Provides a class 
"""

# third party imports
import numpy as np          # matrix support

# standard library imports
import os

# package imports
from ezSCUP.handlers import SP_SCUPHandler, FDFSetting
from ezSCUP.generators import RestartGenerator
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class SPSimulation():
#   - setup(system_name, parameter_file, supercell)
#   - run(generator)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class SPSimulation():


    def setup(self, system_name, parameter_file, supercell):

        self.name = system_name
        self.parameter_file = parameter_file
        self.supercell = supercell

        self.sim = SP_SCUPHandler(self.name, self.parameter_file, cfg.SCUP_EXEC)
        self.sim.settings["supercell"] = [list(self.supercell)]

        pass


    def run(self, generator):

        self.generator = generator
        base_name = "SP-" + self.name
        self.generator.write(base_name + ".restart")
        self.sim.settings["geometry_restart"] =  FDFSetting(base_name + ".restart")
        self.sim.launch(output_file=base_name + ".out")

        os.remove(base_name + ".restart")

        f = open(base_name + ".out")

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
        
        self.energy = energy
        f.close()
        
        return energy