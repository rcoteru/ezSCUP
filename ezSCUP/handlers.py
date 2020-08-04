"""
Collection of classes to handle interaction with ScaleUp.

Allows loading FDF files, to later modify its 
settings and or launch ScaleUp simulations.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports

from ezSCUP.structures import FDFSetting
import ezSCUP.settings as cfg
import ezSCUP.exceptions

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class SCUPHandler()

#####################################################################
## CLASS DEFINITIONS
#####################################################################

class SCUPHandler():

    """

    Loads the setting in a given FDF file, allowing
    easy modification and launch of ScaleUP simulations.
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration .restart file. 
    By default this class starts empty, until a file is loaded with 
    the load() method.

    Basic information about the simulation such as supercell shape,
    number of cells, elements, number of atoms per cell, strain and  
    cell information are accessible via attributes.

    The cell structure in this class only constains displacement info.
    More on cell information storage in ezSCUP.structures.

    """

    fname = ""                  # loaded file name
    scup_exec = ""              # ScaleUP executable path
    settings = None             # Current settings
    original_settings = {}      # Original settings

    #######################################################

    def __init__(self, scup_exec):
        self.scup_exec = scup_exec

    def load(self, fname):
        
        self.fname = fname

        # empty the previous settings
        self.settings = {}
        self.original_settings = {}

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

        self.original_settings = deepcopy(self.settings)
        
        pass




    def save_as(self, fname):

        if self.fname == fname:
            print("WARNING: Attempting to overwite original input. Aborting.")
            return 0

        if len(self.settings) == None:
            print("WARNING: No settings file loaded. Aborting.")
            return 0

        f = open(fname, "w")

        if cfg.PRINT_CONF_SETTINGS:
            print("\nSettings in " + fname + ":")

        for k in self.settings:

            if cfg.PRINT_CONF_SETTINGS:
                print(k, self.settings[k])

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

        if cfg.PRINT_CONF_SETTINGS:
                print("\n")


    def launch(self, output_file=None):

        if len(self.settings) == 0:
            print("WARNING: No settings file loaded.")
            raise ezSCUP.exceptions.InvalidFDFSetting

        if not os.path.exists(self.scup_exec):
            print("WARNING: SCUP executable provided does not exist.")
            raise ezSCUP.exceptions.NoSCUPExecutableDetected


        if output_file == None:
            # set default value after we know settings are loaded
            output_file = self.settings["System_name"] + ".out"

        # create temporary input
        self.save_as("modinput.fdf")

        command = self.scup_exec + " < modinput.fdf > " + output_file

        # execute simulation
        os.system(command)

        # remove temporary input
        os.remove("modinput.fdf")


    def print_all(self):

        for k in self.settings:
            print(k, str(self.settings[k]))

        pass
            