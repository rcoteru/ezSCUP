"""
Collection of auxiliary data structures.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# third party imports
import numpy as np          # matrix support

# package imports
import ezSCUP.settings as cfg

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class Cell()
# class FDFSetting()

#####################################################################
## EXCEPTION DEFINITIONS
#####################################################################

# TO BE IMPLEMENTED

#####################################################################
## DATA STRUCTURES
#####################################################################

class Cell:

    """ 

    Basic container for individual cell data.

    # BASIC USAGE # 
    
    It stores either/both atom position and displacement,
    the number of atoms in the cell and which element they are.

    Positions and displacements are store as Python dictionaries,
    with the keys being the elements name and the value being
    numpy [3x1] arrays. 

    """
 
    nats = 0                    # number of atoms
    elements = []               # elements in the cell, in order
    sc_pos = []                 # position in the supercell
    positions = {}              # dictionary with atomic positions
    displacements = {}          # dictionary with atomic displacements

    #######################################################

    def __init__(self, sc_pos, elements, positions=None, displacements=None):

        """
        Cell class constructor. 

        Parameters:
        ----------

        - sc_pos  (list): position of the cell in the supercell

        - elements (list): element labels

        - pos (dict): dictionary with element labels as keys
            and [3x1] position arrays as values.
        
        - disp (dict): dictionary with element labels as keys
            and [3x1] displacement arrays as values.

        """

        self.nats = len(elements)
        self.elements = elements
        self.sc_pos = sc_pos
        self.positions = positions
        self.displacements = displacements

        pass

    #######################################################

    def __str__(self):
        return "Cell at " + str(self.sc_pos) +  " with " +  str(self.nats) + " atoms."

    #######################################################

    def print_atom_pos(self):

        """Print atomic position information, if available."""

        if self.positions == None:
            print("No position info available.")
        else:
            for atom in self.positions:
                print(atom + str(self.positions[atom]))
        
    #######################################################

    def print_atom_disp(self):

        """Print atomic displacement information, if available."""

        if self.displacements == None:
            print("No displacement info available.")
        else:
            for atom in self.displacements:
                print(atom + str(self.displacements[atom]))

    #######################################################



class FDFSetting():

    """ 

    Defines an standard (non-block) FDF setting.

    # BASIC USAGE # 
    
    Allows storage of both value and unit (id needed)
    for FDF file settings.

    """

    value = 0      # value of the setting
    unit = ""      # unit, if applicable

    #######################################################

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

    #######################################################

    def __str__(self):

        """ Defines conversion to string for printing. """

        if self.unit == "None":
            return str(self.value)
        else:
            return str(self.value) + " " + self.unit

    #######################################################