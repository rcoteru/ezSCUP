"""
Several functions to calculate supercell polarization.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# third party imports
import numpy as np

# package imports
from ezSCUP.simulations import MCConfiguration
from ezSCUP.generators import RestartGenerator
from ezSCUP.geometry import Geometry, UnitCell

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func unit_conversion(scup_polarization)
# + func polarization(config, born_charges)
# + func stepped_polarization(config, born_charges)
# + func layered_polarization(config, born_charges)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def unit_conversion(scup_polarization):

    """
    Converts polarizations from e/bohr2 to C/m2.

    Parameters:
    ----------
    - scup_polarization (float): polarization in e/bohr2

    Return:
    ----------
    - same polarization in C/m2.

    """

    e2C = 1.60217646e-19 # elemental charges to Coulombs
    bohr2m = 5.29177e-11 # bohrs to meters

    return scup_polarization*e2C/bohr2m**2 # polarization in C/m2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def polarization(config, born_charges):

    """

    Calculates supercell polarization in the current configuration
    using the given effective charge vector.

    Parameters:
    ----------
    - born_charges (dict): dictionary with element labels as keys
    and effective charge 3D vectors as values. (in elemental charge units)

    Return:
    ----------
    - a 3D vector with the macroscopic polarization (in C/m2)
    
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    labels = list(born_charges.keys())

    for l in labels:
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    
    cnts = config.geo.lat_constants
    stra = config.geo.strains
    ncells = config.geo.supercell[0]*config.geo.supercell[1]*config.geo.supercell[2]

    ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
    volume = ncells*ucell_volume

    pol = np.zeros(3)
    for x in range(config.geo.supercell[0]):
        for y in range(config.geo.supercell[1]):
            for z in range(config.geo.supercell[2]):

                for label in config.geo.elements:

                    tau = config.geo.cells[x,y,z].displacements[label]
                    charges = born_charges[label]
                    
                    for i in range(3):
                        pol[i] += charges[i]*tau[i]

    pol = pol/volume # in e/bohr2

    return unit_conversion(pol) # in C/m2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def stepped_polarization(config, born_charges):

    """

    Calculates supercell polarization in the current configuration
    for every single .partial file using the given Born effective charges.

    Parameters:
    ----------
    - born_charges (dict): dictionary with element labels as keys
    and effective charge 3D vectors as values. (in elemental charge units)

    Return:
    ----------
    - a list of 3D vectors with the macroscopic polarization (in C/m2)

    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    labels = list(born_charges.keys())

    for l in labels:
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    
    cnts = config.geo.lat_constants
    ncells = config.geo.supercell[0]*config.geo.supercell[1]*config.geo.supercell[2]
    generator = RestartGenerator(config.geo.supercell, config.geo.species, config.geo.nats)

    pol_hist = []
    for f in config.partials:
        
        generator.read(f)

        stra = generator.strains

        ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
        volume = ncells*ucell_volume

        pol = np.zeros(3)
        for x in range(config.geo.supercell[0]):
            for y in range(config.geo.supercell[1]):
                for z in range(config.geo.supercell[2]):

                    for label in config.geo.elements:

                        tau = generator.cells[x,y,z].displacements[label]
                        charges = born_charges[label]
                        
                        for i in range(3):
                            pol[i] += charges[i]*tau[i]

        pol = pol/volume # in e/bohr2
        pol_hist.append(unit_conversion(pol)) # in C/m2
  
    return pol_hist

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def layered_polarization(config, born_charges):

    """

    Calculates supercell polarization in the current configuration
    in horizontal (z-axis) layers using the given effective charges.

    Parameters:
    ----------
    - born_charges (dict): dictionary with element labels as keys
    and effective charge 3D vectors as values. (in elemental charge units)

    Return:
    ----------
    - a list of 3D vectors with the macroscopic polarization (in C/m2)
    
    """

    if not isinstance(config, MCConfiguration):
        raise ezSCUP.exceptions.InvalidMCConfiguration

    labels = list(born_charges.keys())

    for l in labels:
        if l not in config.geo.elements:
            raise ezSCUP.exceptions.InvalidLabel

    
    cnts = config.geo.lat_constants
    stra = config.geo.strains
    ncells_per_layer = config.geo.supercell[0]*config.geo.supercell[1]

    ucell_volume = (cnts[0]*(1+stra[0]))*(cnts[1]*(1+stra[1]))*(cnts[2]*(1+stra[2]))
    volume = ucell_volume*ncells_per_layer

    pols_by_layer = []
    for layer in range(config.geo.supercell[2]):

        pol = np.zeros(3)
        for x in range(config.geo.supercell[0]):
            for y in range(config.geo.supercell[1]):

                for label in config.geo.elements:

                    tau = config.geo.cells[x,y,layer].displacements[label]
                    charges = born_charges[label]
                    
                    for i in range(3):
                        pol[i] += charges[i]*tau[i]

        pol = pol/volume # in e/bohr2
        pols_by_layer.append(unit_conversion(pol))

    return pols_by_layer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


        
