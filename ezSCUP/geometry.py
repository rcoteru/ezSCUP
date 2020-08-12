"""
Collection of auxiliary data structures.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# third party imports
import numpy as np
import csv

# package imports
import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class UnitCell()
#   - __init__(sc_pos, elements, positions, displacements)
#   - print_atom_pos()
#   - print_atom_disp()
#
# + class Geometry()
#   - __init__(reference_file)
#   - load_restart(restart_file)
#   - load_equilibrium_displacements(partials)
#   - write_restart(fname)
#   - write_reference(fname)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class UnitCell:

    """ 

    Basic container for individual unit cell data.

    # BASIC USAGE # 
    

    It stores the basic information regarding a given unit cell,
    namely the species, positions and displacements of its atoms.

    Positions and displacements are stored as Python dictionaries,
    with the keys being the elements name and the value being 
    numpy [3x1] arrays. 

    """
 
    nats = 0                    # number of atoms
    elements = []               # elements in the cell, in order
    sc_pos = []                 # position in the supercell
    positions = {}              # dictionary with atomic positions
    displacements = {}          # dictionary with atomic displacements

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, sc_pos, elements, positions=None, displacements=None):

        """
        Cell class constructor. 

        Parameters:
        ----------

        - sc_pos  (list): position of the cell within the supercell
        - elements (list): element labels, used in the dictionaries.

        - positions (dict): dictionary with element labels as keys
            and [3x1] numpy arrays as values. Contains the reference
            positions of the atoms.
        
        - displacements (dict): dictionary with element labels as keys
            and [3x1] numpy arrays as values. Contains the displacements 
            of the atoms from their reference positions.

        """

        self.nats = len(elements)
        self.elements = elements
        self.sc_pos = sc_pos
        self.positions = positions
        self.displacements = displacements

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __str__(self):
        return "Cell at " + str(self.sc_pos) +  " with " +  str(self.nats) + " atoms."

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def print_atom_pos(self):

        """Print atomic position information, if available."""

        if self.positions == None:
            print("No position info available.")
        else:
            for atom in self.positions:
                print(atom + str(self.positions[atom]))
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def print_atom_disp(self):

        """Print atomic displacement information, if available."""

        if self.displacements == None:
            print("No displacement info available.")
        else:
            for atom in self.displacements:
                print(atom + str(self.displacements[atom]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Geometry():

    """
    Geometry container

    Attributes:
    ----------

     - supercell (array): supercell shape
     - ncells (int): number of unit cells
     - nats (int): number of atoms per unit cell
     - nels (int): number of distinct atomic species
     - species (list): atomic species within the supercell
     - elements (list): labelsfor the atoms within the cells
     
     -strains
     - lat_vectors (1x9 array): lattice vectors, in Bohrs 
     - lat_constants (array): xx, yy, zz lattice constants, in Bohrs

    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, reference_file):

        
        #########
        # CHECK FILE EXISTS
        #########

        f = open(reference_file)

        self.supercell = np.array(list(map(int, f.readline().split())))
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.nats, self.nels = list(map(int, f.readline().split()))
        self.species = f.readline().split()
        self.elements = self.species.copy()

        # adjust final element names
        if self.nats != self.nels:
            for i in range(self.nats-self.nels):
                self.elements.append(self.elements[self.nels-1]+str(2+i))
            self.elements[self.nels-1] = self.elements[self.nels-1]+str(1)

        # read lattice constants
        self.lat_vectors = np.array(list(map(float, f.readline().split())))
        self.lat_constants = np.array([self.lat_vectors[0],self.lat_vectors[4], self.lat_vectors[8]])
        for i in range(self.lat_constants.size): # normalize with supercell size
            self.lat_constants[i] = self.lat_constants[i]/self.supercell[i]

        # create strain placeholder
        self.strains = np.zeros(6)

        # generate cell structure: a UnitCell array
        self.cells = np.zeros(list(self.supercell), dtype="object")
        
        #read reference atomic positions
        for _ in range(self.ncells): # read all unit cells

            sc_pos = []     # current cell in the supercell
            atom_pos = {}   # current cell's atomic positions
            atom_disp = {}  # current cell's atomic displacements

            for j in range(self.nats): # read all atoms within the cell

                line = f.readline().split()
                sc_pos = list(map(int, line[0:3]))
                atom_pos[self.elements[j]] = np.array(list(map(float, line[5:])))
                atom_disp[self.elements[j]] = np.array([0.,0.,0.])

            # store the position in the cell structure
            self.cells[sc_pos[0], sc_pos[1], sc_pos[2]] = UnitCell(
                sc_pos, 
                self.elements, 
                positions=atom_pos, 
                displacements=atom_disp
                )

        f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def reset_geom(self):

        self.strains = np.zeros(6)

        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):

                    for atom in self.cells[x,y,z].elements:
                        self.cells[x,y,z].displacements[atom] = np.zeros(3)

        pass


    def load_restart(self, restart_file):

        """
        Loads a single restart file
        """

        #########
        # CHECK FILE EXISTS
        #########

        self.reset_geom()

        f = open(restart_file)
        
        # checks restart file matches loaded geometry
        rsupercell = np.array(list(map(int, f.readline().split())))
        if not np.all(self.supercell == rsupercell): 
            raise ezSCUP.exceptions.RestartNotMatching()

        rnats, rnels = list(map(int, f.readline().split()))
        if (rnats != self.nats) or (rnels != self.nels):
            raise ezSCUP.exceptions.RestartNotMatching()

        rspecies = f.readline().split()
        if not (set(rspecies) == set(self.species)):
            raise ezSCUP.exceptions.RestartNotMatching()

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
 
        f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def load_equilibrium_displacements(self, partials):

        """
        Obtains the equilibrium (average) displacements
        out of several restart files.
        """
        
        self.reset_geom()

        npartials = len(partials)

        for p in partials: # iterate over all partial .restarts

            #########
            # CHECK FILE EXISTS
            #########

            f = open(p)
        
            # checks restart file matches loaded geometry
            rsupercell = np.array(list(map(int, f.readline().split())))
            if not np.all(self.supercell == rsupercell): 
                raise ezSCUP.exceptions.RestartNotMatching()

            rnats, rnels = list(map(int, f.readline().split()))
            if (rnats != self.nats) or (rnels != self.nels):
                raise ezSCUP.exceptions.RestartNotMatching()

            rspecies = f.readline().split()
            if not (set(rspecies) == set(self.species)):
                raise ezSCUP.exceptions.RestartNotMatching()

            # add strain contributions
            strains += np.array(list(map(float, f.readline().split())))/npartials

            # add displacements contribution
            for _ in range(self.ncells): # read all unit cells

                sc_pos = []     # current cell in the supercell
                atom_disp = {}  # current cell's atomic displacements

                for j in range(self.nats): # read all atoms

                    line = f.readline().split()
                    sc_pos = list(map(int, line[0:3]))
                    atom_disp[self.elements[j]] = np.array(list(map(float, line[5:])))

                self.cells[sc_pos[0], sc_pos[1], sc_pos[2]].displacements += atom_disp/npartials
    
            f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def write_restart(self, fname):

        """ Writes a restart file from loaded data. """

        f = open(fname, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        # write header
        tsv.writerow(list(self.supercell))      
        tsv.writerow([self.nats, self.nels])    
        tsv.writerow(self.species)              
        
        # write strains
        pstrains = list(self.strains)
        pstrains = ["{:.8E}".format(s) for s in pstrains]
        tsv.writerow(pstrains)

        # write displacements
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

    def write_reference(self, fname):

        """ Writes a reference file from loaded data. """

        f = open(fname, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        # write header
        tsv.writerow(list(self.supercell))      
        tsv.writerow([self.nats, self.nels])    
        tsv.writerow(self.species)      
        
        # write lattice vectors
        pvectors = list(self.lat_vectors)
        pvectors = ["{:.8E}".format(s) for s in pvectors]
        tsv.writerow(pvectors) 

        # write positions
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

                        disps = list(cell.positions[self.elements[i]])
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

