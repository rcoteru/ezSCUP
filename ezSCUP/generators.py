"""
Collection of classes to generate various ScaleUp files.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# third party imports
import numpy as np          # matrix support

# standard library imports
from pathlib import Path    # file path management
import os                   # file management
import csv                  # file printing
from copy import deepcopy   # proper array copy

# package imports
import ezSCUP.settings as cfg
from ezSCUP.structures import Cell

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class RestartGenerator()

#####################################################################
## RESTART FILE GENERATOR
#####################################################################

class RestartGenerator:

    """ Generates .restart files to kickstart simulations"""

    supercell = []      # supercell shape
    ncells = 0          # number of cells
    nats = 0            # number of atoms per cell
    nels = 0            # number of distinct atomic elements
    elements = []       # elements in the lattice, in order
    strains = []        # strains, in Voigt notation (percent variation)
    cells = []          # loaded cell data

    #######################################################

    def __init__(self, supercell, elements, nats):

        self.supercell = np.array(supercell)
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.elements = elements.copy()
        self.species = elements.copy()
        self.nels = len(elements)
        self.nats = nats
        self.strains = np.zeros(6)

        # adjust final element names
        if self.nats != self.nels:
            for i in range(self.nats-self.nels):
                self.elements.append(self.elements[self.nels-1]+str(2+i))
            self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        atom_displacement = {}             # empty atomic displacements
        for j in range(self.nats):         # iter over all atoms
            atom_displacement[self.elements[j]] = np.zeros(3)

        self.cells = np.zeros(list(self.supercell), dtype="object")

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    self.cells[x,y,z] = Cell([x, y, z], self.elements, displacements=deepcopy(atom_displacement))
                    
    #######################################################
        
    def write(self, fname):

        f = open(fname, 'wt')
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

    #######################################################

    def print_all(self):

        """ Prints all simulation info available. """

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
## SIMULATION.INFO FILE GENERATOR
#####################################################################

