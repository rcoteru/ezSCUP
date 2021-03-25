"""
Class that provides a data structure to handle SCALE-UP geometry files.
"""

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
# + class Geometry()
#   - __init__(supercell, species, nats)
#   - load_reference(reference_file)
#   - load_restart(restart_file)
#   - load_equilibrium_displacements(partials)
#   - write_restart(restart_file)
#   - write_reference(reference_file)
#   - write_xyz(xyz_file)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Geometry():

    """
    SCALE-UP Geometry Container

    # BASIC USAGE # 

    On creation, the class asks for a supercell shape (ie. [4,4,4]),
    the atomic species in each cell (ie. ["Sr", "Ti", "O"]) and the 
    number of atoms per unit cell (ie. 5). It then creates an empty
    array containing the atomic displacements. Other data must be
    loaded through the class methods.

    Basic information about the simulation such as supercell shape,
    number of cells, elements, number of atoms per cell, lattice
    constants and cell information are accessible via attributes.

    # ACCESSING INDIVIDUAL CELL DATA #

    In order to access the data after loading a file just access either 
    the "positions" or "displacements" attributes in the following manner:

        geo = Geometry(...)                             # instantiate the class
        geo.load_reference("example.REF")               # load a .REF file
        geo.load_reference("example.restart")           # load a .restart file
        geo.positions[x,y,z,j,:]                        # position vector of atom j in cell (x,y,z)
        geo.displacements[x,y,z,j,:]                    # displacement vector of atom j in cell (x,y,z)

    Attributes:
    ----------

     - supercell (array): supercell shape
     - ncells (int): number of unit cells
     - nats (int): number of atoms per unit cell
     - nels (int): number of distinct atomic species
     - species (list): atomic species within the supercell
     - strains (array): supercell strains, in Voigt notation
     - lat_vectors (1x9 array): lattice vectors, in Bohrs 
     - lat_constants (array): xx, yy, zz lattice constants, in Bohrs
     - positions (array): positions of the atoms in the supercell, in Bohrs
     - displacements (array): displacements of the atoms in the supercell, in Bohrs

    """

    B2A = 0.529177 # bohr to angstrom conversion factor

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, supercell, species, nats):
        
        """

        Geometry class constructor.

        Parameters:
        ----------

        - supercell (array): supercell shape (ie. [4,4,4])
        - species (list): atomic species within the supercell in order
        (ie. ["Sr", "Ti", "O"])
        - nats (int): number of atoms per unit cell (ie. 5)

        """

        self.supercell = np.array(supercell)
        self.ncells = int(self.supercell[0]*self.supercell[1]*self.supercell[2])
        self.species = species
        self.nels = len(self.species)
        self.nats = nats

        self.strains = np.zeros(6)

        self.lat_vectors = None
        self.lat_constants = None

        self.positions = None
        sc = self.supercell
        self.displacements = np.zeros([sc[0], sc[1], sc[2], self.nats, 3])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def reset_geom(self):

        """
        Resets all strain and atomic displacement info back to zero.
        """

        self.strains = np.zeros(6)
        sc = self.supercell
        self.displacements = np.zeros([sc[0], sc[1], sc[2], self.nats, 3])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def load_restart(self, restart_file):

        """
        
        Loads the given .restart file's information.

        Parameters:
        ----------

        - restart_file (string): name of the .restart file

        raises: ezSCUP.exceptions.GeometryNotMatching if the geometry contained
        in the .restart file does not match the one loaded from the reference file.

        """

        self.reset_geom()

        f = open(restart_file)
        
        # checks restart file matches loaded geometry
        rsupercell = np.array(list(map(int, f.readline().split())))
        if not np.all(self.supercell == rsupercell): 
            raise ezSCUP.exceptions.GeometryNotMatching()

        rnats, rnels = list(map(int, f.readline().split()))
        if (rnats != self.nats) or (rnels != self.nels):
            raise ezSCUP.exceptions.GeometryNotMatching()

        rspecies = f.readline().split()
        if not (set(rspecies) == set(self.species)):
            raise ezSCUP.exceptions.GeometryNotMatching()

        # read strains 
        self.strains = np.array(list(map(float, f.readline().split())))

        #read displacements
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    # read all atoms within the cell
                    for j in range(self.nats): 
                        line = f.readline().split()
                        self.displacements[x,y,z,j,:] = np.array(list(map(float, line[5:])))

        f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def load_reference(self, reference_file):

        """
        
        Loads the given .REF file's information.

        Parameters:
        ----------

        - reference_file (string): name of the .REF file

        raises: ezSCUP.exceptions.GeometryNotMatching if the geometry contained
        in the .restart file does not match the one loaded from the reference file.

        """

        f = open(reference_file)

        # checks restart file matches loaded geometry
        rsupercell = np.array(list(map(int, f.readline().split())))
        if not np.all(self.supercell == rsupercell): 
            raise ezSCUP.exceptions.GeometryNotMatching()

        rnats, rnels = list(map(int, f.readline().split()))
        if (rnats != self.nats) or (rnels != self.nels):
            raise ezSCUP.exceptions.GeometryNotMatching()

        rspecies = f.readline().split()
        if not (set(rspecies) == set(self.species)):
            raise ezSCUP.exceptions.GeometryNotMatching()

        # read lattice vectors
        self.lat_vectors = np.array(list(map(float, f.readline().split())))
        self.lat_constants = np.array([self.lat_vectors[0],self.lat_vectors[4], self.lat_vectors[8]])
        for i in range(self.lat_constants.size): # normalize with supercell size
            self.lat_constants[i] = self.lat_constants[i]/self.supercell[i]
        self.lat_vectors = np.reshape(self.lat_vectors, (3,3))

        # generate positions array
        sc = self.supercell
        self.positions = np.zeros([sc[0], sc[1], sc[2], self.nats, 3])
        
        #read reference atomic positions
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    # read all atoms within the cell
                    for j in range(self.nats): 
                        line = f.readline().split()
                        self.positions[x,y,z,j,:] = np.array(list(map(float, line[5:])))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def load_equilibrium_displacements(self, partials):

        """
        
        Obtains the equilibrium geometry out of several .restart files
        by averaging out their strains and atomic displacements.

        Parameters:
        ----------

        - partials (list): names of the .restart files.

        raises: ezSCUP.exceptions.RestartNotMatching if the geometry contained in any
        of the .restart file does not match the one loaded from the reference file.

        """
        
        self.reset_geom()

        npartials = len(partials)

        if npartials == 0:
            raise ezSCUP.exceptions.NotEnoughPartials()

        for p in partials: # iterate over all partial .restarts

            f = open(p)
        
            # checks restart file matches loaded geometry
            rsupercell = np.array(list(map(int, f.readline().split())))
            if not np.all(self.supercell == rsupercell): 
                raise ezSCUP.exceptions.GeometryNotMatching()

            rnats, rnels = list(map(int, f.readline().split()))
            if (rnats != self.nats) or (rnels != self.nels):
                raise ezSCUP.exceptions.GeometryNotMatching()

            rspecies = f.readline().split()
            if not (set(rspecies) == set(self.species)):
                raise ezSCUP.exceptions.GeometryNotMatching()

            # add strain contributions
            self.strains += np.array(list(map(float, f.readline().split())))/npartials

            #read displacements
            for x in range(self.supercell[0]):
                for y in range(self.supercell[1]):
                    for z in range(self.supercell[2]):
                        # read all atoms within the cell
                        for j in range(self.nats): 
                            line = f.readline().split()
                            self.displacements[x,y,z,j,:] += np.array(list(map(float, line[5:])))/npartials
    
            f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def write_restart(self, restart_file):

        """ 
        
        Writes a .restart file from the current data. 

        Parameters:
        ----------

        - restart_file (string): .restart geometry file where to write everything.
        WARNING: the file will be overwritten.

        """

        f = open(restart_file, 'wt')
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
                    for j in range(self.nats):
                        
                        line = [x, y ,z, j+1]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        line.append(species)

                        disps = list(self.displacements[x,y,z,j,:])
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

    def write_reference(self, reference_file):

        """ 
        
        Writes a reference (.REF) file from the current data. 

        Parameters:
        ----------

        - reference_file (string): .REF geometry file where to write everything.
        WARNING: the file will be overwritten.

        """

        if self.positions is None: 
            raise ezSCUP.exceptions.PositionsNotLoaded()

        f = open(reference_file, 'wt')
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
    
                    for j in range(self.nats):
                        
                        line = [x, y ,z, j+1]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        line.append(species)

                        disps = list(self.positions[x,y,z,j,:])
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

    def write_xyz(self, xyz_file, comment=".xyz file automatically generated by ezSCUP."):


        if self.positions is None: 
            raise ezSCUP.exceptions.PositionsNotLoaded()

        natoms = self.nats*self.ncells

        strain=np.zeros((3,3))
        for i in range(3):
            strain[i,i]=1+self.strains[i]
        strain[1,2]=self.strains[3]        
        strain[2,1]=self.strains[3]        
        strain[0,2]=self.strains[4]        
        strain[2,0]=self.strains[4]
        strain[0,1]=self.strains[5]        
        strain[1,0]=self.strains[5]

        f = open(xyz_file, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        # write number of atoms and comment
        tsv.writerow([natoms])   
        tsv.writerow([comment])   
        
        # write position of each atom
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
    
                    for j in range(self.nats):
                        
                        line = [x, y ,z, j+1]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        line = [self.species[species-1]]

                        disps = list(np.dot(strain, self.B2A*(self.positions[x,y,z,j,:] + self.displacements[x,y,z,j,:])))
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

    def write_xsf(self, xsf_file, comment=".xsf file automatically generated by ezSCUP.", vector_field=None):

        if self.positions is None: 
            raise ezSCUP.exceptions.PositionsNotLoaded()

        natoms = self.nats*self.ncells

        strain=np.zeros((3,3))
        for i in range(3):
            strain[i,i]=1+self.strains[i]
        strain[1,2]=self.strains[3]        
        strain[2,1]=self.strains[3]        
        strain[0,2]=self.strains[4]        
        strain[2,0]=self.strains[4]
        strain[0,1]=self.strains[5]        
        strain[1,0]=self.strains[5]

        # Get the global cell vectors
        slat_vec = np.dot(strain, self.lat_vectors)
        slat_vec[0,:] = slat_vec[0,:]*self.supercell[0]
        slat_vec[1,:] = slat_vec[1,:]*self.supercell[1] 
        slat_vec[2,:] = slat_vec[2,:]*self.supercell[2]  

        f = open(xsf_file, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        # initial comment
        tsv.writerow(["#", comment])   
        tsv.writerow([])

        tsv.writerow(["CRYSTAL"])
        tsv.writerow(["PRIMVEC"])

        # lattice vectors
        for vec in self.B2A*slat_vec:
            tsv.writerow(["{:8.4f}".format(v) for v in vec])

        tsv.writerow(["PRIMCOORD"])
        tsv.writerow([natoms,1])

        # write position of each atom
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
    
                    for j in range(self.nats):
                        
                        line = [x, y ,z, j+1]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        line = [self.species[species-1]]
                        positions = list(np.dot(strain, self.B2A*(self.positions[x,y,z,j,:] + self.displacements[x,y,z,j,:])))
                        positions = ["{:8.4F}".format(d) for d in positions]

                        line = line + positions

                        if vector_field is not None:
                            vecs = list(vector_field[x,y,z,j,:])
                            vecs = ["{:8.4F}".format(d) for d in vecs]
                            line = line + vecs

                        tsv.writerow(line)

        f.close()

    def write_SIESTA(self, fdf_file):

        if self.positions is None: 
            raise ezSCUP.exceptions.PositionsNotLoaded()

        strain=np.zeros((3,3))
        for i in range(3):
            strain[i,i]=1+self.strains[i]
        strain[1,2]=self.strains[3]        
        strain[2,1]=self.strains[3]        
        strain[0,2]=self.strains[4]        
        strain[2,0]=self.strains[4]
        strain[0,1]=self.strains[5]        
        strain[1,0]=self.strains[5]

        # Get the global cell vectors
        slat_vec = np.dot(strain, self.lat_vectors)
        slat_vec[0,:] = slat_vec[0,:]*self.supercell[0]
        slat_vec[1,:] = slat_vec[1,:]*self.supercell[1] 
        slat_vec[2,:] = slat_vec[2,:]*self.supercell[2]  

        # get positions array
        abs_pos = np.zeros((self.supercell[0], self.supercell[1], self.supercell[2], self.nats, 3))
        frac_pos = np.zeros((self.supercell[0], self.supercell[1], self.supercell[2], self.nats, 3))
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    for j in range(self.nats):
                        abs_pos[x,y,z,j,:]  = np.dot(strain, self.positions[x,y,z,j,:] + self.displacements[x,y,z,j,:])
                        frac_pos[x,y,z,j,0] = abs_pos[x,y,z,j,0]/slat_vec[0,0] 
                        frac_pos[x,y,z,j,1] = abs_pos[x,y,z,j,1]/slat_vec[1,1] 
                        frac_pos[x,y,z,j,2] = abs_pos[x,y,z,j,2]/slat_vec[2,2] 


        f = open(fdf_file, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        tsv.writerow([r"%block AtomicCoordinatesAndAtomicSpecies"])

        # write position of each atom
        at = 0
        for x in range(self.supercell[0]):
            for y in range(self.supercell[1]):
                for z in range(self.supercell[2]):
                    for j in range(self.nats):
                        
                        at += 1
                        line = ["{:10.8F}".format(d) for d in frac_pos[x,y,z,j,:]]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        
                        
                        line += [species]
                        line += [at]
                        line += [self.species[species-1]]

                        tsv.writerow(line)

                        
        tsv.writerow([r"%endblock AtomicCoordinatesAndAtomicSpecies"])
        f.close()

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #
