import xml.etree.ElementTree as ET
import numpy as np
import os

MODEL_FOLDER  = os.getenv("SCUP_MODELS", default = None)
STORED_MODELS = [f for f in os.listdir(MODEL_FOLDER)]

class Model():

    def __init__(self, fname:str):

        self.file = fname
        tree = ET.parse(fname)
        root = tree.getroot()

        # lattice vectors
        self.lat_vecs = np.array([float(c) for c in root.find("unit_cell").text.split()]).reshape([3,3])

        # atom information
        nat = 0
        self.atoms        = {}
        self.ref_struct   = {}
        self.born_charges = {}
        for atom in root.iter("atom"):

            self.atoms[nat] = {
                "species":      atom.attrib.get("element"),
                "mass":         atom.attrib.get("mass"),
                "mass_units":   atom.attrib.get("massunits")
            }

            self.ref_struct[nat] = np.array([float(c) for c in atom.find("position").text.split()])
            self.born_charges[nat] = np.array([float(c) for c in atom.find("borncharge").text.split()]).reshape([3,3])

            nat += 1

        self.nats = 5

        # unit information
        self.length_units = root.find("unit_cell").attrib.get("units")
        self.mass_units   = root.find("atom").attrib.get("massunits")
        self.charge_units = root.find("atom").find("borncharge").attrib.get("units")

        pass

    def reorder(self):

        #TODO

        pass

AVAILABLE_MODELS: dict = {}
for m in STORED_MODELS:
    AVAILABLE_MODELS[m[:-4]] = Model(os.path.join(MODEL_FOLDER, m))

#####################################################################
#                       STO_JPCM2013 MODEL                          #
# https://iopscience.iop.org/article/10.1088/0953-8984/25/30/305401 #
#####################################################################

STO_JPCM2013 = {
    "name": "STO_JPCM2013",                                         # model name
    "file": os.path.join(MODEL_FOLDER, "STO_JPCM2013.xml"),      # model file
    "species": ["Sr", "Ti", "O"],                                   # elements in the lattice
    "masses": [87.6, 47.9, 16, 16, 16],                             # masses, in atomic units
    "labels": [0, 1, 4, 3, 2],                                      # [A, B, 0x, Oy, Oz]
    "nats": 5,                                                      # number of atoms per cell
    "lat_vectors": [                                                # lattice vectors, in bohr
        np.array([7.2655879, 0.0000000, 0.0000000]),
        np.array([0.0000000, 7.2655879, 0.0000000]),
        np.array([0.0000000, 0.0000000, 7.2655879])
    ],
    "born_charges": {                                               # Born effective charges, in e/bohr
        0: np.array([2.566657, 2.566657, 2.566657]),
        1: np.array([7.265894, 7.265894, 7.265894]),
        2: np.array([-2.062603, -2.062603, -5.707345]),
        3: np.array([-2.062603, -5.707345, -2.062603]),
        4: np.array([-5.707345, -2.062603, -2.062603])
    },
    "ref_struct": {                                                 # unit cell atomic structure, in bohr
        0: np.array([0.000000, 0.000000, 0.000000]),
        1: np.array([3.632794, 3.632794, 3.632794]),
        2: np.array([3.632794, 3.632794, 0.000000]),
        3: np.array([3.632794, 0.000000, 3.632794]),
        4: np.array([0.000000, 3.632794, 3.632794])
    },
    "BOdist": 3.632794
}

#####################################################################
#                       STO_PRB2017 MODEL                           #
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.094115  #
#####################################################################

STO_PRB2017 = {
    "name": "STO_PRB2017",                                          # model name
    "file": os.path.join(MODEL_FOLDER, "STO_PRB2017.xml"),       # model file
    "species": ["Ti", "Sr", "O"],                                   # elements in the lattice
    "masses": [47.867, 87.62, 16.0, 16.0, 16.0],                    # masses, in atomic units
    "labels": [1, 0, 2, 3, 4],                                      # [A, B, 0x, Oy, Oz]
    "nats": 5,                                                      # number of atoms per cell
    "lat_vectors": [                                                # lattice vectors, in bohr
        np.array([7.3029865, 0.0000000, 0.0000000]),
        np.array([0.0000000, 7.3029865, 0.0000000]),
        np.array([0.0000000, 0.0000000, 7.3029865]),
    ],
    "born_charges": {                                               # Born effective charges, in e/bohr
        0: [7.333, 7.333, 7.333],
        1: [2.554, 2.554, 2.554],
        2: [-5.765, -2.055, -2.055],
        3: [-2.055, -5.765, -2.055],
        4: [-2.055, -2.055, -2.055]
    },
    "ref_struct": {                                                 # unit cell atomic structure, in bohr
        0: np.array([3.6515000, 3.6515000, 3.6515000]),
        1: np.array([0.0000000, 0.0000000, 0.0000000]),
        2: np.array([0.0000000, 3.6515000, 3.6515000]),
        3: np.array([3.6515000, 0.0000000, 3.6515000]),
        4: np.array([3.6515000, 3.6515000, 0.0000000])
    },
    "BOdist": 3.6515000
}