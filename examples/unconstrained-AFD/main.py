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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# ezSCUP imports
from ezSCUP.simulations import MCSimulation, MCConfiguration, MCSimulationParser
from ezSCUP.analysis import perovskite_AFD, perovskite_simple_rotation
from ezSCUP.files import save_file
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up/build_dir/src/scaleup.x"

OVERWRITE = False                           # overwrite old output folder?

SUPERCELL = [2,2,4]                         # shape of the supercell
ELEMENTS = ["Sr", "Ti", "O"]                # elements in the lattice
NATS = 5                                    # number of atoms per cell
TEMPERATURES = np.linspace(10, 50, 4)       # temperatures to simulate         

cfg.MC_STEPS = 2000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 500            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 50                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 50            # MC steps between output prints  
cfg.FIXED_STRAIN_COMPONENTS = [False]*6     # fixed strain components (none)
cfg.MC_MAX_JUMP = 0.1                       # MC max jump (in Ang, def=0.5)

plot_AFDa =   True                          # plot AFDa distortion angles?
plot_AFDi =   True                          # plot AFDi distortion angles?
plot_strain = True                          # plot strains?
plot_vectors = True                         # plot vector fields?
# temperatures for which to plot rotation vector field
VECTOR_TEMPS = [TEMPERATURES[0]] 

show_plots =  True                          # show the created plots?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


#####################################################################
#              FUNCTIONS TO OBTAIN THE AFD ROTATIONS                #
#####################################################################

def read_AFD(temp, mode="a"):

    """ Calculates the average AFD rotation for each temperature. """

    if mode != "a" and mode != "i":
        raise NotImplementedError

    # create a simulation parser
    sim = MCSimulationParser()

    ###################

    rotations = []
    rotations_err = []
    for t in temp: # read all files

        # Ox, Oy, Oz = O3, O2, O1
        config = sim.access(t)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]

        x_angles, y_angles, z_angles = perovskite_AFD(config, labels, mode)

        x_axis_rot = np.mean(np.abs(x_angles))
        x_axis_rot_err = np.std(np.abs(x_angles))
 
        y_axis_rot = np.mean(np.abs(y_angles))
        y_axis_rot_err = np.std(np.abs(y_angles))

        z_axis_rot = np.mean(np.abs(z_angles))
        z_axis_rot_err = np.std(np.abs(z_angles))

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

    return np.array(rotations), np.array(rotations_err)

def display_AFD(temp, mode="a"):

    """ Generates the AFD distortion graph. """

    # calculating angles from output files
    if mode == "a":
        angles, angles_err = read_AFD(temp, mode="a")
    elif mode == "i":
        angles, angles_err = read_AFD(temp, mode="i")
    else:
        raise NotImplementedError
    
    # unpack the rotations
    xrot = angles[:,0]
    yrot = angles[:,1]
    zrot = angles[:,2]

    xrot_err = angles_err[:,0]
    yrot_err = angles_err[:,1]
    zrot_err = angles_err[:,2]

    headers = ["temp", "xrot", "yrot", "zrot",
        "xrot_err", "yrot_err", "zrot_err"]
    save_file("csv/AFD" + mode + ".csv", headers, 
        [temp, xrot, yrot, zrot, xrot_err, yrot_err, zrot_err])

    # plotting
    
    fig = plt.figure("AFD" + mode + ".png")

    plt.errorbar(temp, xrot, yerr=xrot_err, label=r"AFD$_{x}^{" + mode + "}$", marker ="<") 
    plt.errorbar(temp, yrot, yerr=yrot_err, label=r"AFD$_{y}^{" + mode + "}$", marker =">") 
    plt.errorbar(temp, zrot, yerr=zrot_err, label=r"AFD$_{z}^{" + mode + "}$", marker ="^")
    
    plt.tight_layout(pad = 3)

    plt.xlabel("T (K)", fontsize = 14)
    plt.ylabel(r"$AFD^{" + mode + "}$ (deg)", fontsize = 14)

    plt.ylim(0,8)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)

    plt.savefig("plots/AFD" + mode + ".png")
    plt.draw()

#####################################################################
#              FUNCTIONS TO OBTAIN ROTATION FIELD                   #
#####################################################################


def display_rotation_field(temps):

    """ Generates the rotation vector field graph. """
    
    sim = MCSimulationParser()

    for i, t in enumerate(temps):

        # Ox, Oy, Oz = O3, O2, O1
        config = sim.access(t)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]
        
        u, v ,w = perovskite_simple_rotation(config, labels)
        modules = np.sqrt(np.multiply(u,u) + np.multiply(v,v), np.multiply(w,w))
        nu = np.divide(u, modules)
        nv = np.divide(v, modules)
        nw = np.divide(w, modules)

        x, y, z = np.meshgrid(
            np.linspace(0,int(SUPERCELL[0])-1, SUPERCELL[0]),
            np.linspace(0,int(SUPERCELL[1])-1, SUPERCELL[1]),
            np.linspace(0,int(SUPERCELL[2])-1, SUPERCELL[2]),
        )
    
        fig = plt.figure("vector" + str(i) + ".png", figsize=[8,8])
        ax = fig.gca(projection='3d')
        
        ax.quiver(x, y, z, nu, nv, nw, length=1/np.max(SUPERCELL), lw=3)

        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_zlabel("z", fontsize=14)

        ax.w_xaxis.set_pane_color((1,1,1,0))
        ax.w_yaxis.set_pane_color((1,1,1,0))
        ax.w_zaxis.set_pane_color((1,1,1,0))

        ax.w_xaxis.line.set_color((1,1,1,0))
        ax.w_yaxis.line.set_color((1,1,1,0))
        ax.w_zaxis.line.set_color((1,1,1,0))

        ax.set_xticks(np.linspace(0,int(SUPERCELL[0])-1, SUPERCELL[0]))
        ax.set_yticks(np.linspace(0,int(SUPERCELL[1])-1, SUPERCELL[1]))
        ax.set_zticks(np.linspace(0,int(SUPERCELL[2])-1, SUPERCELL[2]))

        plt.tight_layout(pad = 3)
        plt.grid(False)
        plt.savefig("plots/vector" + str(i) + ".png")
        plt.draw()


#####################################################################
#                 FUNCTIONS TO OBTAIN THE STRAINS                   #
#####################################################################

def read_strain(temps):

    """ Calculates the average strains for each temperature. """

    sim = MCSimulationParser()
    sim.index()

    strains = []
    strains_err = []
    for t in temps: # read all files

        sim.access(t)
        data = sim.parser.lattice_output()

        sx = data["Strn_xx"].mean()
        sy = data["Strn_yy"].mean()
        sz = data["Strn_zz"].mean()

        sx_err = data["Strn_xx"].std()
        sy_err = data["Strn_yy"].std()
        sz_err = data["Strn_zz"].std()
        
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

    """Generates the strains graph."""

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
    save_file("csv/strain.csv", headers, 
        [temps, sx, sy, sz, sx_err, sy_err, sz_err,])

    # plotting 
    plt.figure("strain.png")

    plt.errorbar(temps, sx*100, yerr=sx_err*100, label=r"$\eta_x$", marker ="<") 
    plt.errorbar(temps, sy*100, yerr=sy_err*100, label=r"$\eta_y$", marker =">") 
    plt.errorbar(temps, sz*100, yerr=sz_err*100, label=r"$\eta_z$", marker ="^")

    plt.tight_layout(pad = 3)

    plt.ylabel(r"$\eta$ (%)", fontsize = 14)
    plt.xlabel("T (K)", fontsize = 14)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)
    
    plt.savefig("plots/strain.png")
    plt.draw()


#####################################################################
#                       MAIN FUNCTION CALL                          #
#####################################################################

if __name__ == "__main__":

    # create simulation class
    sim = MCSimulation(SUPERCELL, ELEMENTS, NATS, OVERWRITE)

    # simulate and properly store output
    sim.launch("input.fdf", temp=TEMPERATURES)

    try: #create the "plots" folder if needed
        os.mkdir("plots")
    except FileExistsError:
        pass

    try: # create the "csv" folder if needed
        os.mkdir("csv")
    except FileExistsError:
        pass

    if plot_AFDa: # plot AFDa if needed
        print("\nGenerating AFDa plot...")
        start = time.time()
        display_AFD(TEMPERATURES, mode="a")
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_AFDi: # plot AFDi if needed
        print("\nGenerating AFDi plot...")
        start = time.time()
        display_AFD(TEMPERATURES, mode="i")
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_strain: # plot strain if needed
        print("\nGenerating strain plot...")
        start = time.time()
        display_strain(TEMPERATURES)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_vectors: # plot vector field if needed
        print("\nGenerating strain plot...")
        start = time.time()
        display_rotation_field(VECTOR_TEMPS)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))
   
    if show_plots:
        print("\n Displaying selected plots...")
        plt.show()
    
    print("\nEVERYTHING DONE!")

#####################################################################