"""Example script"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~ REQUIRED MODULE IMPORTS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# standard library imports
import os, sys
import time

# third party imports
import matplotlib.pyplot as plt
import numpy as np

# ezSCUP imports
from ezSCUP.simulations import MCSimulation, MCConfiguration, MCSimulationParser
from ezSCUP.analysis import ModeAnalyzer
from ezSCUP.files import save_file
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up/build_dir/src/scaleup.x"

OVERWRITE = False                           # overwrite old output folder?

SUPERCELL = [3,3,3]                         # shape of the supercell
ELEMENTS = ["Sr", "Ti", "O"]                # elements in the lattice
NATS = 5                                    # number of atoms per cell
TEMPERATURES = np.linspace(10, 300, 1)      # temperatures to simulate

cfg.MC_STEPS = 2000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 500            # MC equilibration steps
cfg.FIXED_STRAIN_COMPONENTS = [False]*6     # fixed strain components (none)

plot_AFDa =   True                          # plot AFDa distortion angles?
plot_AFDi =   False                         # plot AFDi distortion angles?
plot_strain = True                          # plot strains?
show_plots =  True                          # show the created plots?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



#####################################################################
#            FUNCTIONS TO OBTAIN THE AFD ROTATIONS                  #
#####################################################################

def read_AFDa(temps):

    """ Calculates the AFDa distortion rotation for each temperature. """
    
    # Ox, Oy, Oz = O3, O2, O1

    print("Reading files to obtain the AFDa distortion rotation...")

    # create a simulation parser
    sim = MCSimulationParser()

    # create the mode analyzer, so as to
    # project the dirstortions on the lattice
    analyzer = ModeAnalyzer()

    ### DISTORTIONS ###

    AFDa_X=[
            # atom, hopping, weight, target vector
            # "lower" cell
            ["O2",[0, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            ["O1",[0, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            ["O2",[0, 0, 1],1./8.,[ 0.0, 0.0, 1.0]],
            ["O1",[0, 1, 0],1./8.,[ 0.0,-1.0, 0.0]],
            # "upper" cell
            ["O2",[1, 0, 0],1./8.,[ 0.0, 0.0, 1.0]],
            ["O1",[1, 0, 0],1./8.,[ 0.0,-1.0, 0.0]],
            ["O2",[1, 0, 1],1./8.,[ 0.0, 0.0,-1.0]],
            ["O1",[1, 1, 0],1./8.,[ 0.0, 1.0, 0.0]]
        ]

    AFDa_Y=[
            # atom, hopping, weight, target vector
            # lower cell
            ["O3",[0, 0, 0],1./8.,[ 0.0, 0.0, 1.0]],
            ["O1",[0, 0, 0],1./8.,[-1.0, 0.0, 0.0]],
            ["O3",[1, 0, 0],1./8.,[ 0.0, 0.0,-1.0]],
            ["O1",[0, 0, 1],1./8.,[ 1.0, 0.0, 0.0]],
            # upper cell
            ["O3",[0, 1, 0],1./8.,[ 0.0, 0.0,-1.0]],
            ["O1",[0, 1, 0],1./8.,[ 1.0, 0.0, 0.0]],
            ["O3",[1, 1, 0],1./8.,[ 0.0, 0.0, 1.0]],
            ["O1",[0, 1, 1],1./8.,[-1.0, 0.0, 0.0]]
        ]

    AFDa_Z=[
            # atom, hopping, weight, target vector
            # "lower" cell
            ["O3",[0, 0, 0],1./8.,[ 0.0,-1.0, 0.0]],
            ["O2",[0, 0, 0],1./8.,[ 1.0, 0.0, 0.0]],
            ["O3",[1, 0, 0],1./8.,[ 0.0, 1.0, 0.0]],
            ["O2",[0, 1, 0],1./8.,[-1.0, 0.0, 0.0]],
            # "upper" cell
            ["O3",[0, 0, 1],1./8.,[ 0.0, 1.0, 0.0]],
            ["O2",[0, 0, 1],1./8.,[-1.0, 0.0, 0.0]],
            ["O3",[1, 0, 1],1./8.,[ 0.0,-1.0, 0.0]],
            ["O2",[0, 1, 1],1./8.,[ 1.0, 0.0, 0.0]]
        ]

    ###################

    rotations = []
    rotations_err = []
    for t in temps: # read all files

        config = sim.access(t)
        analyzer.load(config)

        cell_zero = config.cells[0,0,0]

        TiO_dist = np.linalg.norm(cell_zero .position["Ti"] - cell_zero.position["03"])

        unit_cell=[[1,0,0],[0,1,0],[0,0,1]]

       
        # X axis rotation   
        distortions = analyzer.measure(SUPERCELL, unit_cell, AFDa_X)
        angles = np.arctan(np.abs(TiO_dist/distortions))*180/np.pi

        x_axis_rot = np.mean(angles)
        x_axis_rot_err = np.std(angles)

        # Y axis rotation   
        distortions = analyzer.measure(SUPERCELL, unit_cell, AFDa_Y)
        angles = np.arctan(np.abs(TiO_dist/distortions))*180/np.pi

        y_axis_rot = np.mean(angles)
        y_axis_rot_err = np.std(angles)

        # Z axis rotation    
        distortions = analyzer.measure(SUPERCELL, unit_cell, AFDa_Z)
        angles = np.arctan(np.abs(TiO_dist/distortions))*180/np.pi

        z_axis_rot = np.mean(angles)
        z_axis_rot_err = np.std(angles)

        rots = np.array([x_axis_rot, y_axis_rot, z_axis_rot])
        rots_err = np.array([x_axis_rot_err, y_axis_rot_err, z_axis_rot_err])
    
        main_axis = np.argmax(rots)

        if main_axis == 2:
            rotations.append(rots)
            rotations_err.append(rots_err)
        elif main_axis == 1:
            rotations.append(np.array([z_axis_rot, x_axis_rot, y_axis_rot]))
            rotations_err.append(np.array([z_axis_rot_err, x_axis_rot_err, y_axis_rot_err]))
        else:
            rotations.append(np.array([y_axis_rot, z_axis_rot, x_axis_rot]))
            rotations_err.append(np.array([y_axis_rot_err, z_axis_rot_err, x_axis_rot_err]))
    
    print("DONE!\n")

    return np.array(rotations), np.array(rotations_err)

def display_AFDa(temps):

    """ Function to generate the AFDa distortion temperature graph. """

    angles, angles_err = read_AFDa(temps) # calculating angles from output files

    print("Plotting angle vs temperature...")

    # unpack the rotations

    xrot = angles[:,0]
    yrot = angles[:,1]
    zrot = angles[:,2]

    xrot_err = angles_err[:,0]
    yrot_err = angles_err[:,1]
    zrot_err = angles_err[:,2]

    headers = ["temp", "xrot", "yrot", "zrot"
        "xrot_err", "yrot_err", "zrot_err"]
    save_file("AFDa.csv", headers, 
        [temps, xrot, yrot, zrot, xrot_err, yrot_err, zrot_err])

    # plotting
    plt.figure("AFDa.png")

    plt.errorbar(temps, xrot, yerr=xrot_err, label=r"AFD$_{x}^{a}$", marker ="<") 
    plt.errorbar(temps, yrot, yerr=yrot_err, label=r"AFD$_{y}^{a}$", marker =">") 
    plt.errorbar(temps, zrot, yerr=zrot_err, label=r"AFD$_{z}^{a}$", marker ="^")
    
    plt.tight_layout(pad = 3)

    plt.xlabel("T (K)", fontsize = 14)
    plt.ylabel(r"AFD (deg)", fontsize = 14)

    plt.ylim(0,8)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)

    plt.savefig("AFDa.png")
    plt.draw()

#####################################################################
#                 FUNCTIONS TO OBTAIN THE STRAINS                   #
#####################################################################

def read_strain(temps):

    sim = MCSimulationParser()
    sim.index()

    strains = []
    strains_err = []
    for t in temps: # read all files

        sim.access(t)
        data = sim.parser.lattice_output()

        sx = data["Strn_xx"].mean()
        sy = data["Strn_yy"].mean()
        sz = data["Strn_yy"].mean()

        sx_err = data["Strn_xx"].std()
        sy_err = data["Strn_yy"].std()
        sz_err = data["Strn_yy"].std()
        
        # rotate to take the z strain as the maximum     
        stra = np.array([sx,sy,sz])
        stra_err = np.array([sx_err,sy_err,sz_err])
        
        main_axis = np.argmax(stra)

        if main_axis == 2:
            strains.append(stra)
            strains_err.append(stra_err)
        elif main_axis == 1:
            strains.append(np.array([sz, sx, sy]))
            strains_err.append(np.array([sz, sx, sy]))
        else:
            strains.append(np.array([sy, sz, sx]))
            strains_err.append(np.array([sy, sz, sx]))

    return np.array(strains), np.array(strains_err)


def display_strain(temps):

    """Generates the strain vs temperature graph."""

    strain, strain_err = read_strain(temps) # reading cell sizes from output files

    # unpacking the cell parameters
    sx = strain[:,0]
    sy = strain[:,1]
    sz = strain[:,2]

    sx_err = strain_err[:,0]
    sy_err = strain_err[:,1]
    sz_err = strain_err[:,2]

    print("c/a = " + str(np.mean((1+sz)/(1+sx))))

    headers = ["temp", "sx", "sy", "sz",
        "sx_err", "sy_err", "sz_err"]
    save_file("cell.csv", headers, 
        [temps, sx, sy, sz, sx_err, sy_err, sz_err,])

    # plotting 
    plt.figure("strain_vs_temp.png")

    plt.errorbar(temps, sx*100, yerr=sx_err, label=r"$\eta_x$", marker ="<") 
    plt.errorbar(temps, sy*100, yerr=sy_err, label=r"$\eta_y$", marker =">") 
    plt.errorbar(temps, sz*100, yerr=sz_err, label=r"$\eta_z$", marker ="^")

    plt.tight_layout(pad = 3)

    plt.ylabel(r"$\eta$ (%)", fontsize = 14)
    plt.xlabel("T (K)", fontsize = 14)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)
    
    plt.savefig("strain_vs_temp.png")
    plt.draw()


#####################################################################
#                       MAIN FUNCTION CALL                          #
#####################################################################

if __name__ == "__main__":

    # create simulation class
    sim = MCSimulation(SUPERCELL, ELEMENTS, NATS, OVERWRITE)

    # simulate and properly store output
    sim.launch("input.fdf", temp=TEMPERATURES)

    if plot_AFDa: # plot AFDa if needed
        print("\nGenerating AFDa plot...")
        start = time.time()
        display_AFDa(TEMPERATURES)
        end = time.time()
        print("\nDONE! Time elapsed: {:d}s".format(end-start))

    if plot_AFDi: # plot AFDi if needed
        print("\nGenerating AFDi plot...")
        start = time.time()
        # display_AFDi(TEMPERATURES) # NOT IMPLEMENTED
        end = time.time()
        print("\nDONE! Time elapsed: {:d}s".format(end-start))

    if plot_strain: # plot strain if needed
        print("\nGenerating strain plot...")
        start = time.time()
        display_strain(TEMPERATURES)
        end = time.time()
        print("\nDONE! Time elapsed: {:d}s".format(end-start))
   
    print("\nDisplaying selected plots...")
    plt.show()
    
    print("\nEVERYTHING DONE!")

#####################################################################