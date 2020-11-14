# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~ REQUIRED MODULE IMPORTS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# third party imports
import numpy as np
import pickle

# ezSCUP imports
from ezSCUP.srtio3.constants import SPECIES, NATS, MASSES
from ezSCUP.normodes import finite_hessian
from ezSCUP.geometry import Geometry
from ezSCUP.singlepoint import SPRun

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [1,1,1]                         # shape of the supercell

# define which coordinates need to be zero in each mode
RESTRICTIONS = [ 
    # translations
    (1, 2, 4, 5, 7, 8, 10, 11, 13, 14),     # x axis only
    (0, 2, 3, 5, 6, 8, 9, 11, 12, 14),      # y axis only
    (0, 1, 3, 4, 6, 7, 9, 10, 12, 13),      # z axis only
    # ferroelectric
    (1, 2, 4, 5, 7, 8, 10, 11, 13, 14),     # x axis only
    (0, 2, 3, 5, 6, 8, 9, 11, 12, 14),      # y axis only
    (0, 1, 3, 4, 6, 7, 9, 10, 12, 13),      # z axis only
    # anti-ferroelectric
    (1, 2, 4, 5, 7, 8, 10, 11, 13, 14),     # x axis only
    (0, 2, 3, 5, 6, 8, 9, 11, 12, 14),      # y axis only
    (0, 1, 3, 4, 6, 7, 9, 10, 12, 13),      # z axis only
    # rotations
    (0, 1, 2, 3, 4, 5, 12, 13, 14),         # x axis rotation
    (0, 1, 2, 3, 4, 5, 9, 10, 11),          # y axis rotation
    (0, 1, 2, 3, 4, 5, 6, 7, 8),            # z axis rotation
    # octahedral deformations
    (1, 2, 4, 5),                           # x axis only
    (0, 2, 3, 5),                           # y axis only
    (0, 1, 3, 4),                           # z axis only
]

run = False                                 # run the hessian calculation? (LONG)
pretty = True                               # print the pretty vectors or the original ones
ortho_check = False                         # print orthogonality check

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

geo = Geometry(SUPERCELL, SPECIES, NATS)

# calculate or load the hessian
if run == True:
    hessian = finite_hessian("srtio3_full_lat.xml", geo, MASSES)
    with open("hessian.pickle", "wb") as f:
                pickle.dump(hessian, f)
else:
    with open("hessian.pickle", "rb") as f:
                hessian = pickle.load(f) 

# create mass matrix
M = []
for m in MASSES:
    for _ in range(3):    
        M.append(m)
M = np.diagflat(M)
Mi = np.linalg.inv(M)

# diagonalize the hessian
normHessian = np.matmul(Mi, hessian)
freqs, modes = np.linalg.eigh(normHessian)

np.set_printoptions(suppress=True, precision=10)

print("")
if pretty:

    ormodes = np.zeros((3*NATS, 3*NATS))
    for space in range(NATS):
        for i in range(3):
            
            res = RESTRICTIONS[3*space + i]
            A = modes[res,space*3:space*3+3]

            B = np.matmul(np.transpose(A), A)
            result = np.linalg.eigh(B)[1][:,0]
            result = result/np.linalg.norm(result)
            
            mode = result[0]*modes[:,space*3] + result[1]*modes[:,space*3+1] + result[2]*modes[:,space*3+2]
            mode = mode/np.linalg.norm(mode)

            ormodes[:,3*space + i] = mode
            
            print("Mode #" + str(3*space + i))
            print("-> Eigenvalue:", freqs[3*space + i])
            print("-> Norm:", np.linalg.norm(mode))
            print("")
            print("Sr: " + str(mode[0:3]))
            print("Ti: " + str(mode[3:6]))
            print("Ox: " + str(mode[12:15]))
            print("Oy: " + str(mode[9:12]))
            print("Oz: " + str(mode[6:9]))
            print("\n---------------------------------------------\n")


    # check orthogonality
    if ortho_check:
        print("\nOrthogonality check:")
        for i in range(3*NATS):
            for j in range(i, 3*NATS):
                print(i, j, np.abs(np.round(np.dot(ormodes[:,i], ormodes[:,j]))))

else:

    for i, val in enumerate(freqs):
        
        print("Mode #" + str(i))
        print("-> Eigenvalue:", val)
        print("-> Norm:", np.linalg.norm(modes[:,i]))
        print("")
        print("Sr: " + str(modes[0:3,i]))
        print("Ti: " + str(modes[3:6,i]))
        print("Oz: " + str(modes[6:9,i]))
        print("Oy: " + str(modes[9:12,i]))
        print("Ox: " + str(modes[12:15,i]))
        print("\n---------------------------------------------\n")

    # check orthogonality
    if ortho_check:
        print("\nOrthogonality check:")
        for i in range(3*NATS):
            for j in range(i, 3*NATS):
                print(i, j, np.abs(np.round(np.dot(ormodes[:,i], ormodes[:,j]))))
