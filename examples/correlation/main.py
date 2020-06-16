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
from ezSCUP.analysis import perovskite_AFD, perovskite_simple_rotation, perovskite_tilting
from ezSCUP.files import save_file
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up/build_dir/src/scaleup.x"

OVERWRITE = False                           # overwrite old output folder?

SUPERCELL = [6,6,1]                         # shape of the supercell
ELEMENTS = ["Sr", "Ti", "O"]                # elements in the lattice
NATS = 5                                    # number of atoms per cell
TEMPERATURES = np.linspace(20, 160, 8)      # temperatures to simulate         

cfg.MC_STEPS = 2000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 500            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 50                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 50            # MC steps between output prints  
cfg.FIXED_STRAIN_COMPONENTS = [False]*6     # fixed strain components (none)
cfg.MC_MAX_JUMP = 0.1                       # MC max jump (in Ang, def=0.5)

HEATMAP_SHAPE = [2,2]                       # shape of the heatmap
SELECTED_TEMPS = TEMPERATURES          # selected temps for plotting

plot_vectors = False                        # plot vector fields?
plot_correlation_heatmap = True             # plot correlation heatmap?
plot_correlation_heatmap_tilting = False    # plot correlation heatmap tilting?
plot_correlation_graph   = False            # plot correlation graph?

show_plots =  True                          # show the created plots?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~ code starts here ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


#####################################################################
#              FUNCTIONS TO HEATMAP CORRELATION                     #
#####################################################################

def display_correlation_heatmap(temps):

    sim = MCSimulationParser()

    sim.access(temps[0])

    correlation=np.zeros((len(temps), sim.supercell[0], sim.supercell[1], sim.supercell[2]))

    for i, t in enumerate(temps):

        config = sim.access(t)
        labels = ["Sr", "Ti", "O3", "O2", "O1"]
        
        xrot, yrot, zrot = perovskite_tilting(config, labels)

        # incluir rotar en funcion de los angulos

        tilting = zrot

        for x0 in range(sim.supercell[0]):
            for y0 in range(sim.supercell[1]):
                for z0 in range(sim.supercell[2]):

                    for X in range(sim.supercell[0]):
                        for Y in range(sim.supercell[1]):
                            for Z in range(sim.supercell[2]):

                                correlation[i,x0,y0,z0] += tilting[x0,y0,z0]*tilting[(x0+X)%sim.supercell[0],
                                                                                   (y0+Y)%sim.supercell[1],
                                                                                   (z0+Z)%sim.supercell[2]]

        norm = np.prod(sim.supercell)
        correlation[i,:,:,:]=np.multiply(1./norm,correlation[i,:,:,:])

        """
        if plot_correlation_heatmap_tilting:

            maxval=0.0
            minval=0.0                
            for x in range(sim.supercell[0]):
                for y in range(sim.supercell[1]):
                    for z in range(sim.supercell[2]):
                        if tilting[x,y,z]>maxval: maxval=tilting[x,y,z]
                        if tilting[x,y,z]<minval: minval=tilting[x,y,z]


            minval=np.min(correlation)
            maxval=np.max(correlation)
            for z in range(sim.supercell[2]):
                plt.figure("plots/heat." + "T{:d}".format(int(t)) + ".plane" + str(z) + ".png")
                plane = correlation[:,:,z].copy()
                plt.contourf(np.transpose(plane),200 ,cmap='Greens')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig("plots/heat." + "T{:d}".format(int(t)) + ".plane" + str(z) + ".png")
                plt.draw()
        """

    minval=np.amin(np.amin(correlation))
    maxval=np.amax(np.amax(correlation))
    for i, t in enumerate(temps):
        for z in range(sim.supercell[2]):
            hor_plane=correlation[i,:,:,z]
            myfig,myaxes = plt.subplots(1,1, figsize=(9,6))
            #csx=myaxes.contourf(np.transpose(hor_plane),np.linspace(minval,maxval,2000) ,cmap='Greens') # cmap='seismic' # Common energy axis
            csx=myaxes.contourf(np.transpose(hor_plane),200 ,cmap='Greens') # cmap='seismic'                              # Individual energy axes
            cbarx = plt.colorbar(csx,ax=myaxes)
            myaxes.set_title(r'C$_x$ (au)',fontsize=20)
            plt.tight_layout()
            plt.savefig('plots/corr_'+str(t)+'_horplane_'+str(z)+'.png')







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

    if plot_correlation_heatmap: # plot correlation heatmap if needed
        print("\nGenerating correlation heatmap plot...")
        start = time.time()
        display_correlation_heatmap(SELECTED_TEMPS)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_correlation_graph: # plot correlation graph if needed
        print("\nGenerating correlation graph plot...")
        start = time.time()
        display_correlation_graph(SELECTED_TEMPS)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))

    if plot_vectors: # plot vector field if needed
        print("\nGenerating strain plot...")
        start = time.time()
        display_rotation_field(SELECTED_TEMPS)
        end = time.time()
        print("\n DONE! Time elapsed: {:.3f}s".format(end-start))
   
    if show_plots:
        print("\n Displaying selected plots...")
        plt.show()
    
    print("\nEVERYTHING DONE!")

#####################################################################