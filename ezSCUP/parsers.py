"""
Collection of classes to parse ScaleUp output files.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Production Ready"

# third party imports
import numpy as np          # matrix support
import pandas as pd         # .out file loading

# standard library imports
from pathlib import Path    # file path management
import os                   # file management
import re                   # regular expressions

# package imports
import ezSCUP.settings as cfg
from ezSCUP.structures import Cell

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class RestartParser()
# class REFParser()
# class OutParser()

#####################################################################
## RESTART FILE PARSER
#####################################################################

class RestartParser:

    """

    Parses all the data from .restart files
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration .restart file. 
    By default this class starts empty, until a file is loaded with 
    the load() method.

    Basic information about the simulation such as supercell shape,
    number of cells, elements, number of atoms per cell, strain and  
    cell information are accessible via attributes.

    # ACCESSING INDIVIDUAL CELL DATA #

    In order to access cell data after loading a file just the "cells" 
    attribute in the following manner:

        parser = RestartParser()                # instantiate the class
        parser.load("example.restart")          # load the target file
        desired_cell = parser.cells[x,y,z]      # access the desired cell
        desired_cell.disp["element_label"]      # access its displacement data by label

    where x, y and z is the position of the desired cell in the supercell.
    This will return a "Cell" class with an attribute "disp", a dictionary
    with the displacement vector for each element label (more within the
    next section)

    More on the Cell class in ezSCUP.structures.

    # ELEMENT LABELING #

    By default, ScaleUp does not label elements in the output beside a 
    non-descript number. This programs assigns a label to every atom in
    order to easily access their data from dictionaries.

    Suppose you have an SrTiO3 cell, only three elements but five atoms.
    Then the corresponding labels would be ["Sr","Ti","O1","O2","O3"].

    Attributes:
    ----------

     - fname (string): loaded file name
     - supercell (array): supercell shape
     - ncells (int): number of cells
     - nats (int): number of atoms per cell
     - nels (int): number of distinct atomic elements
     - elements (list): list of element labels within the cells
     
     - strains (array): strains, in Voigt notation (percent variation)
     - cells (array): loaded cell data (displacements)

    """
    
    #######################################################

    def load(self, fname):

        """
        
        Loads the given .restart file.

        Parameters:
        ----------

        - fname  (string): name of the .restart file

        """


        self.fname = fname
        f = open(fname)

        # reads basic simulation info
        self.supercell = np.array(list(map(int, f.readline().split())))
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.nats, self.nels = list(map(int, f.readline().split()))
        self.elements = f.readline().split()
        self.species = self.elements.copy()

        # adjust final element names
        if self.nats != self.nels:
            for i in range(self.nats-self.nels):
                self.elements.append(self.elements[self.nels-1]+str(2+i))
            self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        # read strains
        self.strains = np.array(list(map(float, f.readline().split())))

        # generate cell structure: list of lists of lists
        self.cells = np.zeros(list(self.supercell), dtype="object")

        #read atom displacements
        for i in range(self.ncells): # read all cells

            # current cell in the supercell
            sc_pos = []  
            
            # displacements of the atoms in the cell
            atom_disps = {}
            
            for j in range(self.nats): # read all atoms

                line = f.readline().split()
                sc_pos = list(map(int, line[0:3]))
                atom_disps[self.elements[j]] = np.array(
                    list(map(float, line[5:]))
                )

            self.cells[sc_pos[0], sc_pos[1], sc_pos[2]] = Cell(sc_pos, self.elements, displacements=atom_disps)

        f.close()

    #######################################################

    def print_all(self):

        """ Prints all available simulation info. """

        print("Simulation data in " + self.fname + "\n")

        print("Supercell shape: " + str(self.supercell))
        print("Atoms per cell: " + str(self.nats))
        print("Elements in cell:")
        print(self.elements)
        print(r"Cell strains: (% change)") 
        print(self.strains)
        print("")
      
        for c in self.cells:
            print(self.cells[c])
            self.cells[c].print_atom_pos()
            self.cells[c].print_atom_disp()
            print("")

    #######################################################

#####################################################################
## REFERENCE FILE PARSER
#####################################################################

class REFParser:

    """

    Parses all the data from .REF files
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration's .REF file. 
    By default this class starts empty, until a file is loaded with 
    the load() method.

    Basic information about the simulation such as supercell shape,
    number of cells, elements, number of atoms per cell, lattice
    constants and cell information are accessible via attributes.

    # ACCESSING INDIVIDUAL CELL DATA #

    In order to access cell data after loading a file just the "cells" 
    attribute in the following manner:

        parser = REFParser()                    # instantiate the class
        parser.load("example.REF")              # load the target file
        desired_cell = parser.cells[x,y,z]      # access the desired cell
        desired_cell.pos["element_label"]       # access its position data by label

    where x, y and z is the position of the desired cell in the supercell.
    This will return a "Cell" class with an attribute "pos", a dictionary
    with the position vector for each element label (more within the
    next section)

    More on the Cell class in ezSCUP.structures.

    # ELEMENT LABELING #

    By default, ScaleUp does not label elements in the output beside a 
    non-descript number. This programs assigns a label to every atom in
    order to easily access their data from dictionaries.

    Suppose you have an SrTiO3 cell, only three elements but five atoms.
    Then the corresponding labels would be ["Sr","Ti","O1","O2","O3"].

    Attributes:
    ----------

     - fname (string): loaded file name
     - supercell (array): supercell shape
     - ncells (int): number of cells
     - nats (int): number of atoms per cell
     - nels (int): number of distinct atomic elements
     - elements (list): list of element labels within the cells
     
     - lat_constants (array): lattice constants, in Voigt notation (Bohrs)
     - cells (array): loaded cell data (reference positions)

    """

    #######################################################

    def load(self, fname):

        """
        
        Loads the given .REF file.

        Parameters:
        ----------

        - fname  (string): name of the .restart file

        """

        self.fname = fname
        f = open(fname)

        # reads basic simulation info
        self.supercell = np.array(list(map(int, f.readline().split())))
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.nats, self.nels = list(map(int, f.readline().split()))
        self.elements = f.readline().split()
        self.species = self.elements.copy()

        # adjust final element names
        if self.nats != self.nels:
            for i in range(self.nats-self.nels):
                self.elements.append(self.elements[self.nels-1]+str(2+i))
            self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        # read lattice constants
        lat_constants = np.array(list(map(float, f.readline().split())))
        self.lat_constants = np.array([lat_constants[0],lat_constants[4], lat_constants[8]])
        for i in range(self.lat_constants.size): # normalize with supercell size
            self.lat_constants[i] = self.lat_constants[i]/self.supercell[i]

         # generate cell structure: list of lists of lists
        self.cells = np.zeros(list(self.supercell), dtype="object")

        #read atom displacements
        for i in range(self.ncells): # read all cells

            # current cell in the supercell
            sc_pos = []  
            
            # positions of the atoms in the cell
            atom_pos = {}
            
            for j in range(self.nats): # read all atoms

                line = f.readline().split()
                sc_pos = list(map(int, line[0:3]))
                atom_pos[self.elements[j]] = np.array(
                    list(map(float, line[5:]))
                )

            self.cells[sc_pos[0], sc_pos[1], sc_pos[2]] = Cell(sc_pos, self.elements, positions=atom_pos)

        f.close()

    #######################################################

    def print_all(self):

        """ Prints all available simulation info. """

        print("Simulation data in " + self.fname + "\n")

        print("Supercell shape: " + str(self.supercell))
        print("Atoms per cell: " + str(self.nats))
        print("Elements in cell:")
        print(self.elements)
        print("Lattice constants: (Bohr)") 
        print(self.lat_constants)
        print("")
      
        for c in self.cells:
            print(self.cells[c])
            self.cells[c].print_atom_pos()
            self.cells[c].print_atom_disp()
            print("")

    #######################################################

#####################################################################
## OUTPUT FILE PARSER
#####################################################################

class OutParser:

    """

    Parses lattice data from output files.
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration's output file. 
    By default this class starts empty, until a file is loaded with 
    the load() method.

    All printed lattice data is accessible via attributes.
    Parsed data is stored as pandas Dataframes. 

    Attributes:
    ----------

     - fname (string): loaded file name
     - lattice_data (Dataframe): loaded lattice data
     - lt_re (string): search string for lattice data
        DEFAULT: ezSCUP.settings.LT_SEARCH_WORD

    """

    #######################################################

    def __init__(self):

        """ OutParser class constructor. """

        self.lt_re = cfg.LT_SEARCH_WORD 

    def load(self, fname):

        """
        
        Loads the given output file.

        Parameters:
        ----------

        - fname  (string): name of the output file

        """

        self.fname = fname

        # create a temporary file that
        # contains only the lattice data
        f = open(fname, "r")
        temp = open("temp.txt", "w")
        for line in f:
            if re.search(self.lt_re, line):
                temp.write(line)
        temp.close()  
        f.close()

        # read the lattice data
        self.lattice_data = pd.read_csv("temp.txt", delimiter=r'\s+')

        # clean it up
        del self.lattice_data["LT:"]
        self.lattice_data.set_index('Iter', inplace=True)

        # remove temporary file
        os.remove("temp.txt")

    #######################################################