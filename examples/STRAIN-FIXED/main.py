"""

Executes STO simulations for three different epitaxial strain restrictions,
temperatures spanning between 20 K and 300 K, in 8x8x8 supercells.


"""

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
from ezSCUP.analysis import perovskite_AFD, perovskite_FE_full, perovskite_FE_simple, BornPolarization
from ezSCUP.files import save_file
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up/build_dir/src/scaleup.x"

OVERWRITE = False                           # overwrite old output folder?

# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [8,8,8]                         # shape of the supercell
ELEMENTS = ["Sr", "Ti", "O"]                # elements in the lattice
NATS = 5                                    # number of atoms per cell

TEMPERATURES = np.linspace(20, 300, 10)     # temperatures to simulate
STRAINS = [                                 # strains to simulate
    [+0.03, +0.03, 0.0, 0.0, 0.0, 0.0],
    [+0.00, +0.00, 0.0, 0.0, 0.0, 0.0],
    [-0.03, -0.03, 0.0, 0.0, 0.0, 0.0]
]   # +-3% and 0% cell strain in the x and y direction (Voigt notation)

cfg.MC_STEPS = 40000                        # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 5000           # MC equilibration steps
cfg.MC_STEP_INTERVAL = 50                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  
# fixed strain components: xx, yy, xy (Voigt notation)
cfg.FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]

cfg.PRINT_CONF_SETTINGS = True              # (DEBUGGING) print individual FDF settings

# ~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

plot_AFDa = True                          # plot AFDa angles?
plot_AFDi = True                          # plot AFDi angles?
FIX_XY_PLANE = True                       # make x rotation always the big one?
plot_strain = True                        # plot strains?

plot_full_FE = True                       # plot B-site full FE?
plot_domain_full_FE = True                # plot domain B-site full FE?
plot_simple_FE_B = True                   # plot B-site simple FE? 
plot_simple_FE_A = True                   # plot A-site simple FE?

plot_polarization = True                  # plot polarization?
plot_domain_polarization = True           # plot polarization with domains?

show_plots =  True                        # show the created plots?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


#####################################################################
#              FUNCTIONS TO OBTAIN THE AFD ROTATIONS                #
#####################################################################

def read_AFD(temp, s, mode="a", threshold=0.):

    """ Calculates the average AFD rotations for each temperature. """
   
    if mode != "a" and mode != "i":
        raise NotImplementedError

    # create a simulation parser
    sim = MCSimulationParser()

    ###################

    rotations = []
    rotations_err = []
    for t in temp: # read all files

        # Ox, Oy, Oz = O3, O2, O1
        config = sim.access(t, s=s)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]

        x_angles, y_angles, z_angles = perovskite_AFD(config, labels, mode)
        
        x_angles = x_angles.flatten()
        x_angles = np.array([rot for rot in x_angles if np.abs(rot) > threshold])

        y_angles = y_angles.flatten()
        y_angles = np.array([rot for rot in y_angles if np.abs(rot) > threshold])

        z_angles = z_angles.flatten()
        z_angles = np.array([rot for rot in z_angles if np.abs(rot) > threshold])

        if x_angles.size == 0:
            x_axis_rot = 0
            x_axis_rot_err = 0
        else:
            x_axis_rot = np.mean(np.abs(x_angles))
            x_axis_rot_err = np.std(np.abs(x_angles))

        if y_angles.size == 0:
            y_axis_rot = 0
            y_axis_rot_err = 0
        else:
            y_axis_rot = np.mean(np.abs(y_angles))
            y_axis_rot_err = np.std(np.abs(y_angles))

        if z_angles.size == 0:
            z_axis_rot = 0
            z_axis_rot_err = 0
        else:
            z_axis_rot = np.mean(np.abs(z_angles))
            z_axis_rot_err = np.std(np.abs(z_angles))

        rots = np.array([x_axis_rot, y_axis_rot, z_axis_rot])
        rots_err = np.array([x_axis_rot_err, y_axis_rot_err, z_axis_rot_err])

        rotations.append(rots)
        rotations_err.append(rots_err)

    return np.array(rotations), np.array(rotations_err)


def display_AFD(temp, strain, mode="a", threshold=0.):

    """ Function to generate the AFD distortion temperature graph. """

    if mode != "a" and mode != "i":
        raise NotImplementedError

    # calculating angles from output files
    data = {}
    for s in strain:
        if mode == "a":
            angles, angles_err = read_AFD(temp, s, mode="a", threshold=threshold)
            data[str(s)] = (angles, angles_err)
        else:
            angles, angles_err = read_AFD(temp, s, mode="i", threshold=threshold)
            data[str(s)] = (angles, angles_err)

    # plotting
    colors = ["black", "blue", "red", "green", "yellow"]
    plt.figure("AFD" + mode + ".png")
    for i, s in enumerate(strain):

        angles, angles_err = data[str(s)]

        # unpack the rotations
        xrot = angles[:,0]
        yrot = angles[:,1]
        zrot = angles[:,2]

        xrot_err = angles_err[:,0]
        yrot_err = angles_err[:,1]
        zrot_err = angles_err[:,2]

        headers = ["temp", "xrot", "yrot", "zrot",
        "xrot_err", "yrot_err", "zrot_err"]
        save_file("csv/AFD" + mode  + str(i) + ".csv", headers, 
            [temp, xrot, yrot, zrot, xrot_err, yrot_err, zrot_err])

        # rotate so that the largest xy plane rotations always point in the x dir (aesthetics)
        if FIX_XY_PLANE:
            for k in range(len(xrot)):

                if xrot[k] > yrot[k]:
                    pass

                else:
                    c = xrot[k]
                    c_err = xrot_err[k]

                    xrot[k] = yrot[k]
                    xrot_err[k] = yrot_err[k]

                    yrot[k] = c
                    yrot_err[k] = c_err

        plt.errorbar(temp, xrot, yerr=xrot_err, c=colors[i], marker ="<")
        plt.errorbar(temp, yrot, yerr=yrot_err, c=colors[i], marker =">")
        plt.errorbar(temp, zrot, yerr=zrot_err, c=colors[i], marker ="^", label="$S_{xy}="+str(s[0])+"$")
    
    plt.tight_layout(pad = 3)

    plt.xlabel("T (K)", fontsize = 14)
    plt.ylabel(r"$AFD_z^{" + mode + "}$ (deg)", fontsize = 14)

    #plt.ylim(0,8)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)

    plt.savefig("plots/AFD" + mode + ".png")
    plt.draw()

#####################################################################
#        FUNCTIONS TO OBTAIN THE FERROELECTRIC DISTORTIONS          #
#####################################################################

def read_simple_FE(temp, s, mode="B"):

    """ Calculates the simple FE distortions for each temperature. """

    if mode != "B" and mode != "A":
        raise NotImplementedError
    
    # create a simulation parser
    sim = MCSimulationParser()

    ###################

    distortions = []
    distortions_err = []
    for t in temp: # read all files

        # Ox, Oy, Oz = O3, O2, O1
        config = sim.access(t, s=s)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]

        x_dist, y_dist, z_dist = perovskite_FE_simple(config, labels, mode)

        # X axis distortion  
        x_axis_dist = np.mean(x_dist)
        x_axis_dist_err = np.std(x_dist)

        # Y axis distortion
        y_axis_dist = np.mean(y_dist)
        y_axis_dist_err = np.std(y_dist)

        # Z axis distortion
        z_axis_dist = np.mean(z_dist)
        z_axis_dist_err = np.std(z_dist)

        dists = np.array([x_axis_dist, y_axis_dist, z_axis_dist])
        dists_err = np.array([x_axis_dist_err, y_axis_dist_err, z_axis_dist_err])

        distortions.append(dists)
        distortions_err.append(dists_err)

    return np.array(distortions), np.array(distortions_err)

def display_simple_FE(temp, strain, mode="B"):

    """ Function to generate the simple FE distortion temperature graph. """

    if mode != "B" and mode != "A":
        raise NotImplementedError

    # calculating distortions from output files
    data = {}
    for s in strain:
        if mode == "B":
            dists, dists_err = read_simple_FE(temp, s, mode="B")
            data[str(s)] = (dists, dists_err)
        else:
            dists, dists_err = read_simple_FE(temp, s, mode="A")
            data[str(s)] = (dists, dists_err)

    # plotting
    colors = ["black", "blue", "red", "green", "yellow"]
    plt.figure("simpleFE_"+ mode +".png")
    for i, s in enumerate(strain):

        dists, dists_err = data[str(s)]

        # unpack the distortions
        xdist = dists[:,0]
        ydist = dists[:,1]
        zdist = dists[:,2]

        xdist_err = dists_err[:,0]
        ydist_err = dists_err[:,1]
        zdist_err = dists_err[:,2]

        headers = ["temp", "xdist", "ydist", "zdist",
        "xdist_err", "ydist_err", "zdist_err"]
        save_file("csv/simpleFE_" + mode + str(i) + ".csv", headers, 
            [temp, xdist, ydist, zdist, xdist_err, ydist_err, zdist_err])

        plt.errorbar(temp, np.abs(xdist), yerr=xdist_err, c=colors[i], marker ="<")
        plt.errorbar(temp, np.abs(ydist), yerr=ydist_err, c=colors[i], marker =">")
        plt.errorbar(temp, np.abs(zdist), yerr=zdist_err, c=colors[i], marker ="^", label="$S_{xy}="+str(s[0])+"$")
    
    plt.tight_layout(pad = 3)

    plt.xlabel("T (K)", fontsize = 14)
    plt.ylabel(r"$\delta_z$ (supercells)", fontsize = 14)

    #plt.ylim(0,8)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)

    plt.savefig("plots/simpleFE_"+ mode +".png")
    plt.draw()


def read_full_FE(temp, s, module=False):

    """ Calculates the FE average distortions for each temperature. """
    
    # create a simulation parser
    sim = MCSimulationParser()

    ###################

    distortions = []
    distortions_err = []
    for t in temp: # read all files

        # Ox, Oy, Oz = O3, O2, O1
        config = sim.access(t, s=s)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]

        x_dist, y_dist, z_dist = perovskite_FE_full(config, labels)

        if module:
            # X axis distortion  
            x_axis_dist = np.mean(np.abs(x_dist))
            x_axis_dist_err = np.std(np.abs(x_dist))

            # Y axis distortion
            y_axis_dist = np.mean(np.abs(y_dist))
            y_axis_dist_err = np.std(np.abs(y_dist))

            # Z axis distortion
            z_axis_dist = np.mean(np.abs(z_dist))
            z_axis_dist_err = np.std(np.abs(z_dist))

        else:
            # X axis distortion  
            x_axis_dist = np.mean(x_dist)
            x_axis_dist_err = np.std(x_dist)

            # Y axis distortion
            y_axis_dist = np.mean(y_dist)
            y_axis_dist_err = np.std(y_dist)

            # Z axis distortion
            z_axis_dist = np.mean(z_dist)
            z_axis_dist_err = np.std(z_dist)

        dists = np.array([x_axis_dist, y_axis_dist, z_axis_dist])
        dists_err = np.array([x_axis_dist_err, y_axis_dist_err, z_axis_dist_err])

        distortions.append(dists)
        distortions_err.append(dists_err)

    return np.array(distortions), np.array(distortions_err)

def display_full_FE(temp, strain, module=False):

    """ Function to generate the FE distortion temperature graph. """

    # calculating angles from output files
    data = {}
    for s in strain:
        if module:
            dists, dists_err = read_full_FE(temp, s, module=True)
            data[str(s)] = (dists, dists_err)
        else:
            dists, dists_err = read_full_FE(temp, s, module=False)
            data[str(s)] = (dists, dists_err)

    # plotting
    colors = ["black", "blue", "red", "green", "yellow"]
    plt.figure("fullFE.png")
    for i, s in enumerate(strain):

        dists, dists_err = data[str(s)]

        # unpack the distortions
        xdist = dists[:,0]
        ydist = dists[:,1]
        zdist = dists[:,2]

        xdist_err = dists_err[:,0]
        ydist_err = dists_err[:,1]
        zdist_err = dists_err[:,2]

        headers = ["temp", "xdist", "ydist", "zdisr",
        "xdist_err", "ydist_err", "zdist_err"]

        if module:
            save_file("csv/abs_fullFE" + str(i) + ".csv", headers, 
                [temp, xdist, ydist, zdist, xdist_err, ydist_err, zdist_err])
        else:
            save_file("csv/fullFE" + str(i) + ".csv", headers, 
                [temp, xdist, ydist, zdist, xdist_err, ydist_err, zdist_err])

        plt.errorbar(temp, np.abs(xdist), yerr=xdist_err, c=colors[i], marker ="<")
        plt.errorbar(temp, np.abs(ydist), yerr=ydist_err, c=colors[i], marker =">")
        plt.errorbar(temp, np.abs(zdist), yerr=zdist_err, c=colors[i], marker ="^", label="$S_{xy}="+str(s[0])+"$")
    
    plt.tight_layout(pad = 3)

    plt.xlabel("T (K)", fontsize = 14)
    plt.ylabel(r"amplitude (bohr)", fontsize = 14)

    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)

    if module:
        plt.savefig("plots/abs_fullFE.png")
    else:
        plt.savefig("plots/fullFE.png")

    plt.draw()


#####################################################################
#                 FUNCTIONS TO OBTAIN THE STRAINS                   #
#####################################################################

def read_strain(temp, s):

    sim = MCSimulationParser()
    sim.index()

    strains = []
    strains_err = []
    for t in temp: # read all files

        sim.access(t, s=s)
        data = sim.parser.lattice_output()

        sx = data["Strn_xx"].mean()
        sy = data["Strn_yy"].mean()
        sz = data["Strn_zz"].mean()

        sx_err = data["Strn_xx"].std()
        sy_err = data["Strn_yy"].std()
        sz_err = data["Strn_zz"].std()
             
        stra = np.array([sx,sy,sz])
        stra_err = np.array([sx_err,sy_err,sz_err])
        
        strains.append(stra)
        strains_err.append(stra_err)

    return np.array(strains), np.array(strains_err)


def display_strain(temp, strain):

    """Generates the strain vs temperature graph."""

    data = {}
    # reading strains from output files
    for s in strain:
        stra, stra_err = read_strain(temp, s) 
        data[str(s)] = (stra, stra_err)
    
    # plotting 
    colors = ["black", "blue", "red", "green", "yellow"]
    plt.figure("strain.png")
    for i, s in enumerate(strain):
        
        stra, stra_err = data[str(s)]

        # unpacking the strains
        sx = stra[:,0]
        sy = stra[:,1]
        sz = stra[:,2]

        sx_err = stra_err[:,0]
        sy_err = stra_err[:,1]
        sz_err = stra_err[:,2]

        headers = ["temp", "sx", "sy", "sz",
        "sx_err", "sy_err", "sz_err"]
        save_file("csv/strain" + str(i) + ".csv", headers, 
            [temp, sx, sy, sz, sx_err, sy_err, sz_err,])

        plt.errorbar(temp, sz*100, yerr=sz_err*100, label="$S_{xy}="+str(s[0])+"$", marker ="^", c=colors[i])

    plt.tight_layout(pad = 3)

    plt.ylabel(r"$\eta_z$ (%)", fontsize = 14)
    plt.xlabel("T (K)", fontsize = 14)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)
    
    plt.savefig("plots/strain.png")
    plt.draw()


#####################################################################
#             FUNCTIONS TO OBTAIN THE POLARIZATION                  #
#####################################################################


def read_polarization(temp, s, module=False):

    sim = MCSimulationParser()
    born = BornPolarization()
    sim.index()

    labels = ["Sr", "Ti", "O3", "O2", "O1"]
    born_charges = {
        "Sr": np.array([2.566657, 2.566657, 2.566657]),
        "Ti": np.array([7.265894, 7.265894, 7.265894]),
        "O3": np.array([-5.707345, -2.062603, -2.062603]),
        "O2": np.array([-2.062603, -5.707345, -2.062603]),
        "O1": np.array([-2.062603, -2.062603, -5.707345]),
    }

    pols = []
    pols_err = []
    for t in temp: # read all files

        config = sim.access(t, s=s)

        born.load(config)
        polx, poly, polz = born.perovs_unit_cell_polarization(born_charges)

        if module:
            px = np.mean(np.abs(polx))
            py = np.mean(np.abs(poly))
            pz = np.mean(np.abs(polz))

            px_err = np.std(np.abs(polx))
            py_err = np.std(np.abs(poly))
            pz_err = np.std(np.abs(polz))
        else:
            px = np.mean(polx)
            py = np.mean(poly)
            pz = np.mean(polz)

            px_err = np.std(polx)
            py_err = np.std(poly)
            pz_err = np.std(polz)
           
        pol = np.array([px,py,pz])
        pol_err = np.array([px_err,py_err,pz_err])
        
        pols.append(pol)
        pols_err.append(pol_err)

    return np.array(pols), np.array(pols_err)


def display_polarization(temp, strain, module=False):

    """Generates the polarization vs temperature graph."""

    data = {}
    # reading polarization from output files
    for s in strain:
        if module:
            pol, pol_err = read_polarization(temp, s, module=True) 
            data[str(s)] = (pol, pol_err)
        else:
            pol, pol_err = read_polarization(temp, s, module=False) 
            data[str(s)] = (pol, pol_err)
    
    # plotting 
    colors = ["black", "blue", "red", "green", "yellow"]
    plt.figure("polarization.png")
    for i, s in enumerate(strain):
        
        pol, pol_err = data[str(s)]

        # unpacking the strains
        px = pol[:,0]
        py = pol[:,1]
        pz = pol[:,2]

        px_err = pol_err[:,0]
        py_err = pol_err[:,1]
        pz_err = pol_err[:,2]

        headers = ["temp", "px", "py", "pz",
        "px_err", "py_err", "pz_err"]

        if module:
            save_file("csv/abs_polarization" + str(i) + ".csv", headers, 
                [temp, px, py, pz, px_err, py_err, pz_err,])
        else:
            save_file("csv/polarization" + str(i) + ".csv", headers, 
                [temp, px, py, pz, px_err, py_err, pz_err,])

        plt.errorbar(temp, np.abs(px), yerr=px_err, marker ="<", c=colors[i])
        plt.errorbar(temp, np.abs(py), yerr=py_err, marker =">", c=colors[i])
        plt.errorbar(temp, np.abs(pz), yerr=pz_err, marker ="^", c=colors[i], label="$S_{xy}="+str(s[0])+"$")

    plt.tight_layout(pad = 3)

    plt.ylabel(r"$P$ (C/m2)", fontsize = 14)
    plt.xlabel("T (K)", fontsize = 14)
    
    plt.legend(frameon = True, fontsize = 14)
    plt.grid(True)
    
    if module:
        plt.savefig("plots/abs_polarization.png")
    else:
        plt.savefig("plots/polarization.png")

    plt.draw()

#####################################################################
#                       MAIN FUNCTION CALL                          #
#####################################################################

if __name__ == "__main__":

    # create simulation class
    sim = MCSimulation(SUPERCELL, ELEMENTS, NATS, OVERWRITE)

    # simulate and properly store output
    sim.launch("input.fdf", temp=TEMPERATURES, strain=STRAINS)

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
        display_AFD(TEMPERATURES, STRAINS, mode="a", threshold=1)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_AFDi: # plot AFDi if needed
        print("\nGenerating AFDi plot...")
        start = time.time()
        display_AFD(TEMPERATURES, STRAINS, mode="i", threshold=0.5)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_full_FE: # plot full B-site FE if needed
        print("\nGenerating B-site full FE plot...")
        start = time.time()
        display_full_FE(TEMPERATURES, STRAINS, module=True)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_domain_full_FE: # plot full B-site domain FE if needed
        print("\nGenerating domain B-site full FE plot...")
        start = time.time()
        display_full_FE(TEMPERATURES, STRAINS, module=False)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_simple_FE_B: # plot simple B-site FE if needed
        print("\nGenerating B-site simple FE plot...")
        start = time.time()
        display_simple_FE(TEMPERATURES, STRAINS, mode="B")
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_simple_FE_A: # plot simple A-site FE if needed
        print("\nGenerating A-site simple FE plot...")
        start = time.time()
        display_simple_FE(TEMPERATURES, STRAINS, mode="A")
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_strain: # plot strain if needed
        print("\nGenerating strain plot...")
        start = time.time()
        display_strain(TEMPERATURES, STRAINS)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_polarization: # plot polarization if needed
        print("\nGenerating polarization plot...")
        start = time.time()
        display_polarization(TEMPERATURES, STRAINS, module=True)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_domain_polarization: # plot polarization if needed
        print("\nGenerating domain polarization plot...")
        start = time.time()
        display_polarization(TEMPERATURES, STRAINS, module=False)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))
   
    if show_plots:
        print("\n Displaying selected plots...")
        plt.show()

    print("\nEVERYTHING DONE!")

#####################################################################