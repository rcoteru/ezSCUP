
from ezSCUP.models import Model
import numpy as np
import csv


class Geometry():

    # conversion factors
    bohr2ang = 0.529177             # bohr to angstrom
    ang2bohr = 1.8897259886         # anstrom to bohr
    e2C      = 1.60217646e-19       # elemental charge to Coulomb
    bohr2m   = 5.29177e-11          # bohrs to meters

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, sc: list, model: Model):

        self.sc             = np.array(sc)
        self.model          = model
        self.atoms          = model.atoms 
        self.nats           = model.nats
        self.ref_lat_vecs   = model.lat_vecs

        self.species = []
        for at in self.atoms.keys():
            species = model.atoms[at]["species"]
            if species not in self.species:
                self.species.append(species)

        self.strain         = np.identity(3)
        self.born_cherges   = model.born_charges
        self.reference      = np.zeros((sc[0], sc[1], sc[2], self.nats, 3))
        self.displacements  = np.zeros((sc[0], sc[1], sc[2], self.nats, 3))

        for x in range(self.sc[0]):
            for y in range(self.sc[1]):
                for z in range(self.sc[2]):
                    cell_vec = x*self.ref_lat_vecs[0,:] + y*self.ref_lat_vecs[1,:] + z*self.ref_lat_vecs[2,:]
                    for j in range(self.nats):
                        self.reference[x,y,z,j,:] = cell_vec + model.ref_struct[j]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def reset(self):

        sc = self.sc
        self.strain         = np.identity(3)
        self.displacements  = np.zeros((sc[0], sc[1], sc[2], self.nats, 3))


    def set_voigt_strain(self, str_voigt:list):

        # read strains 
        str_voigt = np.array(np.array(str_voigt))
        
        self.strain = np.identity(3)

        for i in range(3):
            self.strain[i,i] += str_voigt[i]
        
        self.strain[1,2] = str_voigt[3]
        self.strain[2,1] = str_voigt[3]

        self.strain[0,2] = str_voigt[4]
        self.strain[2,0] = str_voigt[4]

        self.strain[0,1] = str_voigt[5]
        self.strain[1,0] = str_voigt[5]


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #                    SCALE-UP RESTART FILES                     #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def read_restart(self, restart_file: str):

        self.reset()

        with open(restart_file, "r") as f:
        
            # checks restart file matches loaded geometry
            rsc = np.array(list(map(int, f.readline().split())))
            if not np.all(self.sc == rsc): 
                print("ERROR: Supercell does not match")

            rnats, rnels = list(map(int, f.readline().split()))
            if (rnats != self.nats) or (rnels != len(self.species)):
                print("ERROR: Number of atoms/elements does not match.")

            rspecies = f.readline().split()
            if not (set(rspecies) == set(self.species)):
                print("ERROR: Atomic species dont match")

            # read strains 
            str_voigt = np.array(list(map(float, f.readline().split())))
            for i in range(3):
                self.strain[i,i] += str_voigt[i]
            
            self.strain[1,2] = str_voigt[3]
            self.strain[2,1] = str_voigt[3]

            self.strain[0,2] = str_voigt[4]
            self.strain[2,0] = str_voigt[4]

            self.strain[0,1] = str_voigt[5]
            self.strain[1,0] = str_voigt[5]

            #read displacements
            for x in range(self.sc[0]):
                for y in range(self.sc[1]):
                    for z in range(self.sc[2]):
                        # read all atoms within the cell
                        for j in range(self.nats): 
                            line = f.readline().split()
                            self.displacements[x,y,z,j,:] = np.array(list(map(float, line[5:])))


    def write_restart(self, restart_file: str):


        with open(restart_file, "w") as f:

            tsv = csv.writer(f, delimiter="\t")

            # write header
            tsv.writerow(list(self.sc))      
            tsv.writerow([self.nats, len(self.species)])    
            tsv.writerow(self.species)              
        
            # write strains
            st = self.strain - np.identity(3)
            str_voigt = [st[0,0], st[1,1], st[2,2], st[1,2], st[0,2], st[0,1]]
            fstrains = ["{:.8E}".format(s) for s in str_voigt]
            tsv.writerow(fstrains)

            # write displacements
            for x in range(self.sc[0]):
                for y in range(self.sc[1]):
                    for z in range(self.sc[2]):
                        for j in range(self.nats):
                            
                            line = [x, y ,z, j+1]

                            if j+1 > len(self.species):
                                spe = len(self.species)
                            else:
                                spe = j+1
                            
                            line.append(spe)

                            disps = list(self.displacements[x,y,z,j,:])
                            disps = ["{:.8E}".format(d) for d in disps]

                            line = line + disps
                        
                            tsv.writerow(line)
            
            f.close()

    def read_restart_average(self, partials: list):

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
        for p in partials: # iterate over all partial .restarts

            f = open(p)
        
            # checks restart file matches loaded geometry
            rsc = np.array(list(map(int, f.readline().split())))
            if not np.all(self.sc == rsc): 
                print("ERROR: Supercell does not match")

            rnats, rnels = list(map(int, f.readline().split()))
            if (rnats != self.nats) or (rnels != self.nels):
                print("ERROR: Number of atoms/elements does not match.")

            rspecies = f.readline().split()
            if not (set(rspecies) == set(self.species)):
                print("ERROR: Atomic species dont match")

            # add strain contributions
            self.strains += np.array(list(map(float, f.readline().split())))/npartials

            #read displacements
            for x in range(self.sc[0]):
                for y in range(self.sc[1]):
                    for z in range(self.sc[2]):
                        # read all atoms within the cell
                        for j in range(self.nats): 
                            line = f.readline().split()
                            self.displacements[x,y,z,j,:] += np.array(list(map(float, line[5:])))/npartials
    
            f.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #                      EXTERNAL FILES                           #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def write_xyz(self, xyz_file, comment=".xyz file automatically generated by ezSCUP."):

        natoms = self.nats*self.ncells

        f = open(xyz_file, 'wt')
        tsv = csv.writer(f, delimiter="\t")

        # write number of atoms and comment
        tsv.writerow([natoms])   
        tsv.writerow([comment])   
        
        # write position of each atom
        for x in range(self.sc[0]):
            for y in range(self.sc[1]):
                for z in range(self.sc[2]):
    
                    for j in range(self.nats):
                        
                        line = [x, y ,z, j+1]

                        if j+1 > self.nels:
                            species = self.nels
                        else:
                            species = j+1
                        
                        line = [self.species[species-1]]

                        disps = list(np.dot(self.strain, self.B2A*(self.positions[x,y,z,j,:] + self.displacements[x,y,z,j,:])))
                        disps = ["{:.8E}".format(d) for d in disps]

                        line = line + disps
                    
                        tsv.writerow(line)
        
        f.close()

    def write_xsf(self, xsf_file, comment=".xsf file automatically generated by ezSCUP.", vector_field=None):

        natoms = self.nats*self.ncells

        # Get the global cell vectors
        slat_vec = np.dot(self.strain, self.lat_vectors)
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
                        positions = list(np.dot(self.strain, self.B2A*(self.positions[x,y,z,j,:] + self.displacements[x,y,z,j,:])))
                        positions = ["{:8.4F}".format(d) for d in positions]

                        line = line + positions

                        if vector_field is not None:
                            vecs = list(vector_field[x,y,z,j,:])
                            vecs = ["{:8.4F}".format(d) for d in vecs]
                            line = line + vecs

                        tsv.writerow(line)

        f.close()


# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #
