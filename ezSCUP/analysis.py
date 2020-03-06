"""
Collection of classes for general analysis of output data.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# third party imports
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.simulations import MCConfiguration
from ezSCUP.structures import FDFSetting, Cell
import ezSCUP.settings as cfg
import ezSCUP.exceptions

#####################################################################
## MODULE STRUCTURE
#####################################################################

# class ModeProjector()
# func octahedra_rotation()

#####################################################################
## MODE ANALYZER
#####################################################################

class ModeAnalyzer():

    """ TODO DOCUMENTATION"""

    config = None    
    
    supercell = []

    new_supercell = []
    variable = []
    unit_cell = []

    def load(self, config):

        """TODO DOCUMENTATION"""

        if not isinstance(config, MCConfiguration):
            raise ezSCUP.exceptions.InvalidMCConfiguration

        self.config = config

        self.supercell = config.supercell

    #######################################################

    def measure(self, new_sc, unit_cell, variable):

        """TODO DOCUMENTATION"""

        self.new_sc = new_sc
        self.unit_cell = unit_cell
        self.variable = variable

        values = np.zeros(new_sc)

        for x in range(new_sc[0]):
            for y in range(new_sc[1]):
                for z in range(new_sc[2]):

                    cell = np.dot([x,y,z], self.supercell)

                    for atom in self.variable:

                        atom_cell = np.mod(cell + atom[1], self.supercell)

                        nx, ny, nz = atom_cell

                        values[x,y,z] += atom[2]*np.dot(atom[3],
                            self.config.cells[nx,ny,nz].displacements[atom[0]])

        self.values = values
        return values

        #######################################################

        


#####################################################################
## FUNCTION DEFINITIONS
#####################################################################

def octahedra_rotation(cell, no, nx, ny ,nz):

    """
        Calculates the rotation of the octahedron in the given cell.

        Parameters:
        ----------

        - cell  (Cell): cell in which to perform the calculation

        - no (string): name of the central atom

        - nx (string): name of the atom in the x direction

        - ny (string): name of the atom in the y direction

        - nz (string): name of the atom in the z direction

    """

    # check if a valid cell was given
    if not isinstance(cell, Cell):
            raise ezSCUP.exceptions.InvalidCell

    # check if all labels are valid
    labels = [no, nx, ny, nz]
    for l in labels:
        if l not in cell.elements:
            raise ezSCUP.exceptions.InvalidLabel("Label " + str(l) + " unrecognized.")

    def normalize(v):
        return v/np.linalg.norm(v)

    def to_degrees(v):
        return v*180./np.pi

    ti_old = cell.positions[no]
    
    ox_old = cell.positions[nx] - ti_old
    oy_old = cell.positions[ny] - ti_old
    oz_old = cell.positions[nz] - ti_old

    ti_new = cell.positions[no] + cell.displacements[no]
 
    ox_new = cell.positions[nx] + cell.displacements[nx] - ti_new
    oy_new = cell.positions[ny] + cell.displacements[ny] - ti_new
    oz_new = cell.positions[nz] + cell.displacements[nz] - ti_new

    # x axis rotation ---------------------------
    # yz plane projection

    yz_oy_old = normalize(oy_old[1:])
    yz_oy_new = normalize(oy_new[1:])
    oy_rot = np.arccos(np.dot(yz_oy_old, yz_oy_new))

    yz_oz_old = normalize(oz_old[1:])
    yz_oz_new = normalize(oz_new[1:])
    oz_rot = np.arccos(np.dot(yz_oz_old, yz_oz_new))

    x_axis_rot = to_degrees(np.mean([oy_rot, oz_rot]))

    # y axis rotation ---------------------------
    # xz plane projection

    xz_ox_old = normalize(ox_old[::2])
    xz_ox_new = normalize(ox_new[::2])
    ox_rot = np.arccos(np.dot(xz_ox_old, xz_ox_new))

    xz_oz_old = normalize(oz_old[::2])
    xz_oz_new = normalize(oz_new[::2])
    oz_rot = np.arccos(np.dot(xz_oz_old, xz_oz_new))

    y_axis_rot = to_degrees(np.mean([ox_rot, oz_rot]))
  
    # z axis rotation ---------------------------
    # xy plane projection

    xy_oy_old = normalize(oy_old[:-1])
    xy_oy_new = normalize(oy_new[:-1])
    oy_rot = np.arccos(np.dot(xy_oy_old, xy_oy_new))

    xy_ox_old = normalize(ox_old[:-1])
    xy_ox_new = normalize(ox_new[:-1])
    ox_rot = np.arccos(np.dot(xy_ox_old, xy_ox_new))

    z_axis_rot = to_degrees(np.mean([oy_rot, ox_rot]))

    return np.array([x_axis_rot, y_axis_rot, z_axis_rot])