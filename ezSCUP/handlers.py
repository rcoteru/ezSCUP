"""
Classes and structures to handle direct interaction with SCALE-UP,
from FDF settings to launching simulations.
"""

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + FDFSetting()
#   - __init__()
#   - __str__()
#
# + class SCUPHandler()
#   - __init__()
#   - load()
#   - save_as()
#   - launch()
#   - print_all()
# 
# + class MC_SCUPHandler(SCUPHandler)
#   - __init__()
#
# + class SP_SCUPHandler(SCUPHandler)
#   - __init__()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class FDFSetting():

    """ 

    Structure for an standard (non-block) FDF setting.    
    Allows easy storage of both value and units (if needed).

    Attributes:
    ----------
    
    - value (int, float, boolean): value of the setting
    - unit (string): unit of the setting, if applicable

    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, value, unit=None):

        """
        
        FDFSetting class constructor.

        Parameters:
        ----------
        - value  (?): value of the setting
        - unit (string): unit for the setting, if needed.

        """

        self.value = value
        self.unit = str(unit).strip()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __str__(self):

        """ Defines conversion to string for printing. """

        if self.unit == "None":
            return str(self.value)
        else:
            return str(self.value) + " " + self.unit

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

class SCUPHandler():

    """

    Loads the setting in a given FDF file, allowing easy input 
    modification and launch of SCALE-UP simulations.
    
    # BASIC USAGE # 
    
    # FDF SETTING STORAGE
    All settings are stored in the settings dictionary.
    Non-block settings are stored as FDFSetting objects,
    while 

    Attributes:
    ----------

    - scup_exec (string): path to SCALE-UP executable
    - fname (string): loaded input file
    - settings (dict): current FDF settings

    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, system_name, parameter_file, scup_exec):

        """
        SCUPHandler class constructor.

        Parameters:
        ----------
        - scup_exec (string): path to the system's SCALE-UP executable

        """

        self.scup_exec = scup_exec
        
        self.settings = {}
        self.settings["system_name"] = FDFSetting(system_name)
        self.settings["parameter_file"] = FDFSetting(parameter_file)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def load(self, fname):

        """
        Loads an SCALE-UP input (.fdf) file, storing its settings for
        a future simulation. Removes all previous settings in the process.

        Parameters:
        ----------
        - fname (string): input (.fdf) file to load.
        
        """
        
        self.fname = fname

        # empty the previous settings
        self.settings = {}

        f = open(fname, "r")
        flines = f.readlines()

        # strip all the useless stuff
        flines = [l.strip() for l in flines]         # remove trailing spaces
        flines = [l.split("#",1)[0] for l in flines] # removes comments
        flines = [l for l in flines if l]            # remove empty lines
        flines = [l.lower() for l in flines]         # make everything lowercase

        # separate setting elements
        flines = [l.split() for l in flines]         # split lines into lists

        i = 0
        block = None
        blockname = None

        while i < len(flines): 
            
            line = flines[i]
            
            # check if its a block setting
            if line[0] == r"%block":

                block = []
                blockname = line[1]
                
                i+=1
                line = flines[i]

                while line[0] != r"%endblock":
                    block.append(line)
                    i += 1
                    line = flines[i]

                self.settings[blockname] = block

            # if its not, then just read it
            else:
                if len(line) == 1:
                    # setting without value = true
                    self.settings[line[0]] = FDFSetting(".true.")

                if len(line) == 2:
                    self.settings[line[0]] = FDFSetting(line[1])

                if len(line) == 3:
                    self.settings[line[0]] = FDFSetting(line[1], unit=line[2])

            i += 1
        
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def save_as(self, fname):

        """
        Save the current FDF settings in self.settings as an SCALE-UP
        input file (fdf).

        Parameters:
        ----------
        - fname (string): filepath where to save the input file.
        WARNING: existing file with the same name may be overwritten.
        
        """

        if len(self.settings) == None:
            print("WARNING: No settings file loaded. Aborting.")
            return 0

        f = open(fname, "w")

        for k in self.settings:

            if type(self.settings[k]) == FDFSetting:
                line = k + " " +  str(self.settings[k]) + "\n"
                f.write(line)

            else:
                f.write(r"%block " + k + "\n")

                # turn array into string
                string = ""
                for row in self.settings[k]:
                    for n in row:
                        string += str(n) + " "
                    string += "\n"

                f.write(string)
                f.write(r"%endblock " + k + "\n")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def launch(self, output_file=None):

        """
        Execute a SCALE-UP simulation with the current FDF settings.

        Parameters:
        ----------
        - output_file: human output filename. Defaults to [system_name].out.

        """

        if not os.path.exists(self.scup_exec):
            print("WARNING: SCUP executable provided does not exist.")
            raise ezSCUP.exceptions.NoSCUPExecutableDetected


        if output_file == None:
            # set default value after we know settings are loaded
            output_file = self.settings["System_name"] + ".out"

        # create temporary input
        self.save_as("_ezSCUPmoddedinput.fdf")

        command = self.scup_exec + " < _ezSCUPmoddedinput.fdf > " + output_file

        # execute simulation
        os.system(command)

        # remove temporary input
        os.remove("_ezSCUPmoddedinput.fdf")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def print(self):

        """
        Print current FDF settings.
        """

        for k in self.settings:
            print(k, str(self.settings[k]))

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
            
# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #


class MC_SCUPHandler(SCUPHandler):


    def __init__(self, system_name, parameter_file, scup_exec):

        super().__init__(system_name, parameter_file, scup_exec)
        
        # general settings
        self.settings["no_electron"] = FDFSetting(".true.")
        self.settings["supercell"] = [[1, 1, 1]]
        self.settings["print_std_lattice_nsteps"] = FDFSetting(cfg.LATTICE_OUTPUT_INTERVAL)

        # MC settings:
        self.settings["run_mode"] = FDFSetting("monte_carlo")
        self.settings["mc_strains"] = FDFSetting(".true.")

        self.settings["mc_temperature"] = FDFSetting(10, unit="kelvin")
        self.settings["mc_annealing_rate"] = FDFSetting(cfg.MC_ANNEALING_RATE)

        self.settings["mc_nsweeps"] = FDFSetting(cfg.MC_STEPS)
        self.settings["mc_max_step"] = FDFSetting(cfg.MC_MAX_JUMP, unit="ang")

        self.settings["print_justgeo"] = FDFSetting(".true.")
        self.settings["n_write_mc"] = FDFSetting(cfg.MC_STEP_INTERVAL)

        # strain settings 
        if len(cfg.FIXED_STRAIN_COMPONENTS) != 6:
            raise ezSCUP.exceptions.InvalidFDFSetting
        auxsetting = []
        for s in list(cfg.FIXED_STRAIN_COMPONENTS):
            if not isinstance(s, bool):
                raise ezSCUP.exceptions.InvalidFDFSetting
            if s:
                auxsetting.append("T")
            else:
                auxsetting.append("F")
        self.settings["fix_strain_component"] = [auxsetting]

        # output file values
        self.settings["print_std_energy"] = FDFSetting(".true.")
        self.settings["print_std_av_energy"] = FDFSetting(".true.")
        self.settings["print_std_delta_energy"] = FDFSetting(".true.")

        self.settings["print_std_polarization"] = FDFSetting(".true.")
        self.settings["print_std_av_polarization"] = FDFSetting(".true.")

        self.settings["print_std_strain"] = FDFSetting(".true.")
        self.settings["Print_std_av_strain"] = FDFSetting(".true.")

        self.settings["print_std_temperature"] = FDFSetting(".true.")


        pass

    pass

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #


class SP_SCUPHandler(SCUPHandler):


    def __init__(self, system_name, parameter_file, scup_exec):

        super().__init__(system_name, parameter_file, scup_exec)
        
        # general settings
        self.settings["no_electron"] = FDFSetting(".true.")
        self.settings["supercell"] = [[1, 1, 1]]

        # SP settings:
        self.settings["run_mode"] = FDFSetting("single_point")

        # strain settings 
        if len(cfg.FIXED_STRAIN_COMPONENTS) != 6:
            raise ezSCUP.exceptions.InvalidFDFSetting
        auxsetting = []
        for s in list(cfg.FIXED_STRAIN_COMPONENTS):
            if not isinstance(s, bool):
                raise ezSCUP.exceptions.InvalidFDFSetting
            if s:
                auxsetting.append("T")
            else:
                auxsetting.append("F")
        self.settings["fix_strain_component"] = [auxsetting]

        pass

    pass