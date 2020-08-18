"""
Classes to generate geometry (.restart) and simulation (simulation.info) files.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# third party imports
import numpy as np          # matrix support

# standard library imports
from pathlib import Path    # file path management
from copy import deepcopy   # proper array copy
import os, csv              # file management


# package imports
from ezSCUP.geometry import UnitCell

import ezSCUP.settings as cfg
import ezSCUP.exceptions


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class RestartGenerator()
#   - __init__(supercell, species, nats)
#   - read(restart_file, overwrite=False)
#   - write(restart_file)
#   - print_all() 
#
# + class SimulationFileGenerator() # TODO
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class RestartGenerator:

    """ Generates .restart files to kickstart SCALE-UP simulations.

    # BASIC USAGE # 

    On creation, the class asks for a supercell shape (ie. [4,4,4]),
    the atomic species in each cell (ie. ["Sr", "Ti", "O"]) and the 
    number of atoms per unit cell (ie. 5). It then creates a cell
    structure within the self.cells attribute, containing displacements
    from an unspecified reference structure (they are all zero on creation).
    This attribute may be modified via external access in order to obtain 
    the desired structure.

    The self.read() and self.write() methods provide ways to respectively 
    load and create .restart geometry files for SCALE-UP using this information.

    # ELEMENT LABELING #

    By default, SCALE-UP does not label elements in the output beside a 
    non-descript number. This programs assigns a label to every atom in
    order to easily access their data from dictionaries.

    In the case of the SrTiO3 cell, you have only three elements but five atoms.
    The corresponding labels would then be ["Sr","Ti","O1","O2","O3"].
    
    Attributes:
    ----------

     - supercell (array): supercell shape
     - ncells (int): number of unit cells
     - nats (int): number of atoms per unit cell
     - nels (int): number of distinct atomic species
     - species (list): atomic species within the supercell
     - elements (list): labels for the atoms within the cells
     - strains (array): supercell strains, in Voigt notation
     - cells (array): array of UnitCell objects (ezSCUP.geometry.UnitCell)
     containing the displacements from the reference geometry, by default zero.
 
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, supercell, species, nats):

        """
        RestartGenerator class constructor.

        Parameters:
        ----------

        - supercell (array): supercell shape (ie. [4,4,4])
        - species (list): atomic species within the supercell 
        (ie. ["Sr", "Ti", "O"])
        - - nats (int): number of atoms per unit cell (ie. 5)

        """

        self.supercell = np.array(supercell)
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.elements = species.copy()
        self.species = species.copy()
        self.nels = len(species)
        self.nats = nats
        self.strains = np.zeros(6)

        # adjust final element names
        if self.nats != self.nels:
            for i in range(self.nats-self.nels):
                self.elements.append(self.elements[self.nels-1]+str(2+i))
            self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        atom_displacement = {}             # empty atomic displacements
        for j in range(self.nats):         # iterate over all atoms
            atom_displacement[self.elements[j]] = np.zeros(3)

        self.cells = np.zeros(list(self.supercell), dtype="object")

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    self.cells[x,y,z] = UnitCell([x, y, z], self.elements, displacements=deepcopy(atom_displacement))
                    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def reset(self):

        """
        Resets all strain and atomic displacement info back to zero.
        """

        self.strains = np.zeros(6)

        atom_displacement = {}             # empty atomic displacements
        for j in range(self.nats):         # iterate over all atoms
            atom_displacement[self.elements[j]] = np.zeros(3)

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    self.cells[x,y,z] = UnitCell([x, y, z], self.elements, displacements=deepcopy(atom_displacement))
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def read(self, restart_file, overwrite=False):

        """
        Load the geometry information from the given .restart file. 

        Parameters:
        ----------

        - restart_file (string): .restart geomtry file where to read from
        - overwrite (boolean): whether to respect the geometry defined on
        the generator's creation (default) or just accept the file's one.


        raises: ezSCUP.exceptions.RestartNotMatching if overwrite is disabled
        and the geometry from file and object does not match.

        """

        self.reset()

        f = open(restart_file)
        
        rsupercell = np.array(list(map(int, f.readline().split())))
        rnats, rnels = list(map(int, f.readline().split()))
        rspecies = f.readline().split()

        if overwrite == False: # checks restart file matches previous geometry

            if not np.all(self.supercell == rsupercell): 
                raise ezSCUP.exceptions.RestartNotMatching()
            if (rnats != self.nats) or (rnels != self.nels):
                raise ezSCUP.exceptions.RestartNotMatching()
            if not (set(rspecies) == set(self.species)):
                raise ezSCUP.exceptions.RestartNotMatching()

        else: # just load the file, ignoring original settings

            self.supercell == rsupercell
            self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
            self.nats = rnats
            self.nels = rnels
            self.species = rspecies
            self.elements = rspecies.copy()

            # adjust final element names
            if self.nats != self.nels:
                for i in range(self.nats-self.nels):
                    self.elements.append(self.elements[self.nels-1]+str(2+i))
                self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        # read strains 
        self.strains = np.array(list(map(float, f.readline().split())))

        # read atom displacements
        for _ in range(self.ncells): # read all unit cells

            sc_pos = []     # current cell in the supercell
            atom_disp = {}  # current cell's atomic displacements

            for j in range(self.nats): # read all atoms

                line = f.readline().split()
                sc_pos = list(map(int, line[0:3]))
                atom_disp[self.elements[j]] = np.array(list(map(float, line[5:])))

            self.cells[sc_pos[0], sc_pos[1], sc_pos[2]].displacements=atom_disp

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def write(self, restart_file):

        """
        Writes the geometry information to a given .restart file. 

        Parameters:
        ----------

        - restart_file (string): .restart geometry file where to write everything.
        WARNING: the file will be overwritten.

        """

        f = open(restart_file, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        tsv.writerow(list(self.supercell))      # supercell
        tsv.writerow([self.nats, self.nels])    # number of atoms, elements
        tsv.writerow(self.species)              # element species
        
        pstrains = list(self.strains)
        pstrains = ["{:.8E}".format(s) for s in pstrains]
        tsv.writerow(pstrains)                  # strains

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
 
                    cell = self.cells[x,y,z]
    
                    for i in range(self.nats):
                        
                        line = [x, y ,z, i+1]

                        if i+1 > self.nels:
                            species = self.nels
                        else:
                            species = i+1
                        
                        line.append(species)

                        disps = list(cell.displacements[self.elements[i]])
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def print_all(self):

        """ Prints all geometry info available. """

        print("Supercell shape: " + str(self.supercell))
        print("Atoms per cell: " + str(self.nats))
        print("Elements in cell:")
        print(self.elements)
        print(r"Cell strains: (% change)") 
        print(self.strains)
        print("")
      
        for c in self.cells:
            print(self.cells[c])
            self.cells[c].print_atom_disp()
            print("")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class SimulationFileGenerator():

    # TODO

    pass