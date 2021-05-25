from ezSCUP.structures import Geometry
from ezSCUP.models import Model
import numpy as np


class STOGeometry(Geometry):

    def __init__(self, sc:list, model:Model):

        super().__init__(sc, model)

        Sr_id = -1
        Ti_id = -1
        for at in model.atoms.keys():
            if model.atoms[at]['species'] == "Sr":
                Sr_id = at 
            if model.atoms[at]['species'] == "Ti":
                Ti_id = at

        self.id_list = [Sr_id, Ti_id]

        Odirs: dict = {}
        for at in model.atoms.keys():
            if model.atoms[at]['species'] == "O":
                Odirs[at] = np.argmax(np.abs(model.ref_struct[Ti_id] - model.ref_struct[at]))

        for i in range(3):
            dir = -1
            for at in Odirs.keys():
                if Odirs[at] == i:
                    dir = at
            self.id_list.append(dir)

    def rotations(self, angles:bool = True):

        _, B, Ox, Oy, Oz = self.id_list

        ROT_X=[
            # atom, hopping, weight, target vector
            [Oz, [ 0, 0, 0],  1/4., [ 0.0, 1.0, 0.0]],
            [Oy, [ 0, 0, 0], -1/4., [ 0.0, 0.0, 1.0]],
            [Oz, [ 0, 0, 1], -1/4., [ 0.0, 1.0, 0.0]],
            [Oy, [ 0, 1, 0],  1/4., [ 0.0, 0.0, 1.0]],
        ]

        ROT_Y=[
            # atom, hopping, weight, target vector
            [Ox, [ 0, 0, 0],  1/4.,[ 0.0, 0.0, 1.0]],
            [Oz, [ 0, 0, 0], -1/4.,[ 1.0, 0.0, 0.0]],
            [Ox, [ 1, 0, 0], -1/4.,[ 0.0, 0.0, 1.0]],
            [Oz, [ 0, 0, 1],  1/4.,[ 1.0, 0.0, 0.0]],
        ]

        ROT_Z=[
            # atom, hopping, weight, target vector
            [Ox, [ 0, 0, 0], -1/4., [ 1.0, 0.0, 0.0]],
            [Oy, [ 0, 0, 0],  1/4., [ 0.0, 1.0, 0.0]],
            [Ox, [ 1, 0, 0],  1/4., [ 0.0, 1.0, 0.0]],
            [Oy, [ 0, 1, 0], -1/4., [ 1.0, 0.0, 0.0]],
        ]
        
        rots  = np.zeros((self.sc[0], self.sc[1], self.sc[2], 3))
        disps = self.displacements

        for x in range(self.sc[0]):
            for y in range(self.sc[1]):
                for z in range(self.sc[2]):

                    cell = np.array([x,y,z])
                
                    for atom in ROT_X:
                        atom_cell = np.mod(cell + atom[1], self.sc)
                        nx, ny, nz = atom_cell
                        rots[x,y,z,0] += atom[2]*np.dot(atom[3], disps[nx,ny,nz,atom[0],:])

                    for atom in ROT_Y:
                        atom_cell = np.mod(cell + atom[1], self.sc)
                        nx, ny, nz = atom_cell
                        rots[x,y,z,1] += atom[2]*np.dot(atom[3], disps[nx,ny,nz,atom[0],:])

                    for atom in ROT_Z:
                        atom_cell = np.mod(cell + atom[1], self.sc)
                        nx, ny, nz = atom_cell
                        rots[x,y,z,2] += atom[2]*np.dot(atom[3], disps[nx,ny,nz,atom[0],:])

        if angles:
            BOd = np.linalg.norm(self.model.ref_struct[B] - self.model.ref_struct[Ox])
            return np.arctan(rots/BOd)*180/np.pi
        else:
            return rots
                
    def polarization(self):

        A, B, Ox, Oy, Oz = self.id_list

        B_centered_pattern = [  # atom, hopping, weight
            # "frame"
            [A, [0, 0, 0], 1./8.],
            [A, [1, 0, 0], 1./8.],
            [A, [1, 1, 0], 1./8.],
            [A, [0, 1, 0], 1./8.],
            [A, [0, 0, 1], 1./8.],
            [A, [1, 0, 1], 1./8.],
            [A, [1, 1, 1], 1./8.],
            [A, [0, 1, 1], 1./8.],
            # "octahedra"
            [B,  [0, 0, 0], 1.   ], # b site
            [Ox, [0, 0, 0], 1./2.], 
            [Ox, [1, 0, 0], 1./2.],
            [Oy, [0, 0, 0], 1./2.],
            [Oy, [0, 1, 0], 1./2.],
            [Oz, [0, 0, 0], 1./2.],
            [Oz, [0, 0, 1], 1./2.]
        ]

        strained_ref_vecs = np.dot(self.strain, self.ref_lat_vecs)
        uc_vol = np.prod(np.linalg.norm(strained_ref_vecs, axis=0))
        pols = np.zeros((self.sc[0], self.sc[1], self.sc[2], 3))
        disps = self.displacements

        for x in range(self.sc[0]):
            for y in range(self.sc[1]):
                for z in range(self.sc[2]):

                    cell = np.array([x,y,z])

                    for atom in B_centered_pattern:

                        atom_cell = np.mod(cell + atom[1], self.sc)
                        nx, ny, nz = atom_cell

                        charge = np.array(self.model.born_charges[atom[0]])
                        pols[x,y,z,:] += atom[2]*np.dot(disps[nx,ny,nz,atom[0],:], charge)

        pols = pols/uc_vol                    # in e/bohr2
        pols = pols*self.e2C/self.bohr2m**2   # in C/m2

        return pols

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#               functions to create specific geometries                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #     

def STO_AFD(sc: list, model: Model, angle:float, 
            mode:str="a", axis:str="z", clockwise:bool=False):

    geom = STOGeometry(sc, model)
    _, B, Ox, Oy, Oz = geom.id_list

    BOd = np.linalg.norm(model.ref_struct[B] - model.ref_struct[Ox])
    disp = BOd*np.sin(angle/180.*np.pi)

    if clockwise:
        cw = 1
    else:
        cw = 0

    for x in range(sc[0]):
        for y in range(sc[1]):
            for z in range(sc[2]):

                if mode == "a":

                    factor = (-1)**x * (-1)**y * (-1)**z * (-1)**cw

                    if axis == "x":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                    elif axis == "y":
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                    elif axis == "z":
                        geom.displacements[x,y,z,Ox,1] -= factor*disp[1]
                        geom.displacements[x,y,z,Oy,0] += factor*disp[0]
                    elif axis == "xy" or axis == "yx":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                    else:
                        raise NotImplementedError()

                elif mode == "i":

                    if axis == "x":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                    elif axis == "y":
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                    elif axis == "z":
                        factor = (-1)**x * (-1)**y * (-1)**cw
                        geom.displacements[x,y,z,Ox,1] -= factor*disp[1]
                        geom.displacements[x,y,z,Oy,0] += factor*disp[0]
                    elif axis == "xy" or axis == "yx":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                    else:
                        raise NotImplementedError()
    
                else:
                    raise NotImplementedError()

    return geom


def STO_FE(sc:list, model:Model, disp:float, axis:str="z"):

    geom = STOGeometry(sc, model)
    _, B, _, _, _ = geom.id_list

    for x in range(sc[0]):
        for y in range(sc[1]):
            for z in range(sc[2]):

                if axis == "x":
                    geom.displacements[x,y,z,B,0] += disp
                elif axis == "y":
                    geom.displacements[x,y,z,B,1] += disp
                elif axis == "z":
                    geom.displacements[x,y,z,B,2] += disp
                elif axis == "xy" or axis == "yx":
                    geom.displacements[x,y,z,B,0] += disp
                    geom.displacements[x,y,z,B,1] += disp
                else:
                    raise NotImplementedError()

    return geom


def STO_AFD_FE(sc:list, model:Model, angle:float, ti_disp:float, 
                mode:str = "a", axis:str = "z", clockwise:bool = False):

    geom = STOGeometry(sc, model)
    _, B, Ox, Oy, Oz = geom.id_list

    BOd = np.linalg.norm(model.ref_struct[B] - model.ref_struct[Ox])
    disp = BOd*np.sin(angle/180.*np.pi)

    if clockwise:
        cw = 1
    else:
        cw = 0

    for x in range(sc[0]):
        for y in range(sc[1]):
            for z in range(sc[2]):

                if mode == "a":

                    factor = (-1)**x * (-1)**y * (-1)**z * (-1)**cw

                    if axis == "x":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        geom.displacements[x,y,z,B,0]  += ti_disp
                    elif axis == "y":
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    elif axis == "z":
                        geom.displacements[x,y,z,Ox,1] -= factor*disp
                        geom.displacements[x,y,z,Oy,0] += factor*disp
                        geom.displacements[x,y,z,B,2]  += ti_disp
                    elif axis == "xy" or axis == "yx":
                        geom.displacements[x,y,z,Oy,2] -= factor*disp
                        geom.displacements[x,y,z,Oz,1] += factor*disp
                        geom.displacements[x,y,z,Oz,0] -= factor*disp
                        geom.displacements[x,y,z,Ox,2] += factor*disp
                        geom.displacements[x,y,z,B,0]  += ti_disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    else:
                        raise NotImplementedError()

                elif mode == "i":

                    if axis == "x":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        geom.displacements[x,y,z,B,0]  += ti_disp
                    elif axis == "y":
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    elif axis == "z":
                        factor = (-1)**x * (-1)**y * (-1)**cw
                        geom.displacements[x,y,z,Ox,1] -= factor*disp[1]
                        geom.displacements[x,y,z,Oy,0] += factor*disp[0]
                        geom.displacements[x,y,z,B,2]  += ti_disp
                    elif axis == "xy" or axis == "yx":
                        factor = (-1)**y * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oy,2] -= factor*disp[2]
                        geom.displacements[x,y,z,Oz,1] += factor*disp[1]
                        factor = (-1)**x * (-1)**z * (-1)**cw
                        geom.displacements[x,y,z,Oz,0] -= factor*disp[0]
                        geom.displacements[x,y,z,Ox,2] += factor*disp[2]
                        geom.displacements[x,y,z,B,0]  += ti_disp
                        geom.displacements[x,y,z,B,1]  += ti_disp
                    else:
                        raise NotImplementedError()
    
                else:
                    raise NotImplementedError()

    return geom

def STO_AFD_FE_XY_DOM(sc:list, model:Model, angle:float, Ti_disp:float, region_size):

    geom = STOGeometry(sc, model)
    _, B, Ox, Oy, Oz = geom.id_list

    BOd = np.linalg.norm(model.ref_struct[B] - model.ref_struct[Ox])
    disp = BOd*np.sin(angle/180.*np.pi)

    for x in range(sc[0]):
        for y in range(sc[1]):
            for z in range(sc[2]):

                factor = (-1)**x * (-1)**y * (-1)**z

                if np.floor(x/region_size)%2 == 0:
                    if np.floor(y/region_size)%2 == 0:
                        case = "A"
                    else:
                        case = "B"
                else:
                    if np.floor(y/region_size)%2 == 0:
                        case = "C"
                    else:
                        case = "D"

                if case == "A":
                    # rotación x negativa
                    geom.displacements[x,y,z,Oy,2] += factor*disp
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    # rotación y
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    # pol x positivo y negativo
                    geom.displacements[x,y,z,B,0]  -= Ti_disp
                    geom.displacements[x,y,z,B,1]  += Ti_disp

                if case == "B":
                    # rotación x negativa
                    geom.displacements[x,y,z,Oy,2] += factor*disp
                    geom.displacements[x,y,z,Oz,1] -= factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,0] += factor*disp
                    geom.displacements[x,y,z,Ox,2] -= factor*disp
                    # pol x negativo y negativo
                    geom.displacements[x,y,z,B,0]  += Ti_disp
                    geom.displacements[x,y,z,B,1]  += Ti_disp

                if case == "C":
                    # rotación x
                    geom.displacements[x,y,z,Oy,2] -= factor*disp
                    geom.displacements[x,y,z,Oz,1] += factor*disp
                    # rotación y
                    geom.displacements[x,y,z,Oz,0] -= factor*disp
                    geom.displacements[x,y,z,Ox,2] += factor*disp
                    # pol x positivo y positivo
                    geom.displacements[x,y,z,B,0]  -= Ti_disp
                    geom.displacements[x,y,z,B,1]  -= Ti_disp

                if case == "D":
                    # rotación x 
                    geom.displacements[x,y,z,Oy,2] -= factor*disp
                    geom.displacements[x,y,z,Oz,1] += factor*disp
                    # rotación y negativa
                    geom.displacements[x,y,z,Oz,0] += factor*disp
                    geom.displacements[x,y,z,Ox,2] -= factor*disp
                    # pol x negativo y positivo
                    geom.displacements[x,y,z,B,0]  += Ti_disp
                    geom.displacements[x,y,z,B,1]  -= Ti_disp
                    
    return geom
