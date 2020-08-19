"""

Generates the graphs presented in the thesis 
from the simulation data created after running "main.py"

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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# ezSCUP imports
from ezSCUP.simulations import MCSimulation, MCConfiguration, MCSimulationParser
from ezSCUP.perovskite import perovskite_polarization, perovskite_simple_rotation
from ezSCUP.perovskite import perovskite_AFD, perovskite_FE_full
from ezSCUP.files import save_file
import ezSCUP.settings as cfg

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~ USER DEFINED SETTINGS ~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# IMPORTANT: location of the Scale-Up executable in the system
cfg.SCUP_EXEC = "/home/raul/Software/scale-up/build_dir/src/scaleup.x"

OVERWRITE = False                           # overwrite old output folder?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~ SIMULATION SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

SUPERCELL = [6,6,6]                         # shape of the supercell
SPECIES = ["Sr", "Ti", "O"]                 # elements in the lattice
LABELS = ["Sr", "Ti", "O3", "O2", "O1"]     # [A, B, 0x, Oy, Oz]
NATS = 5                                    # number of atoms per cell
BORN_CHARGES = {                            # Born effective charges
        "Sr": np.array([2.566657, 2.566657, 2.566657]),
        "Ti": np.array([7.265894, 7.265894, 7.265894]),
        "O3": np.array([-5.707345, -2.062603, -2.062603]),
        "O2": np.array([-2.062603, -5.707345, -2.062603]),
        "O1": np.array([-2.062603, -2.062603, -5.707345]),
    }

TEMPERATURES = np.linspace(20, 400, 15)     # temperatures to simulate
STRAINS = [                                 # strains to simulate
    [+0.03, +0.03, 0.0, 0.0, 0.0, 0.0],
    [+0.00, +0.00, 0.0, 0.0, 0.0, 0.0],
    [-0.03, -0.03, 0.0, 0.0, 0.0, 0.0]
]   # +-3% and 0% cell strain in the x and y direction (Voigt notation)


cfg.MC_STEPS = 1000                         # MC total steps
cfg.MC_EQUILIBRATION_STEPS = 100            # MC equilibration steps
cfg.MC_STEP_INTERVAL = 20                   # MC steps between partial files
cfg.LATTICE_OUTPUT_INTERVAL = 10            # MC steps between output prints  
# fixed strain components: xx, yy, xy (Voigt notation)
cfg.FIXED_STRAIN_COMPONENTS = [True, True, False, False, False, True]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

plot_expansivo_0 = True
plot_expansivo_1 = True
plot_expansivo_1_extra = True
plot_expansivo_2 = True
plot_expansivo_7 = True

plot_compresivo_1 = True
plot_3Dcompresivo_1 = True

show_plots =  True                          # show the created plots?

LABELS = ["Sr", "Ti", "O3", "O2", "O1"]     # A, B, Ox, Oy, Oz

BORN_CHARGES = {
        "Sr": np.array([2.566657, 2.566657, 2.566657]),
        "Ti": np.array([7.265894, 7.265894, 7.265894]),
        "O3": np.array([-5.707345, -2.062603, -2.062603]),
        "O2": np.array([-2.062603, -5.707345, -2.062603]),
        "O1": np.array([-2.062603, -2.062603, -5.707345]),
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                     MAIN FUNCTION CALL                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":

    try: #create the "plots" folder if needed
        os.mkdir("FINAL_PLOTS")
    except FileExistsError:
        pass


    sim = MCSimulationParser()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dominios expansivo plano xy T=20K ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_expansivo_0:

        config = sim.access(TEMPERATURES[0], s=STRAINS[0])

        plt.figure("expansivo_dominios0", figsize=[16,5])
        plt.subplots_adjust(left=0.03, bottom=0.12, right=0.99, top=0.9, wspace=0.02, hspace=0)
        plt.tight_layout()

        polx, poly, polz = perovskite_polarization(config, LABELS, BORN_CHARGES)

        polu = polx[:,:,0]
        polv = poly[:,:,0]

        print("x polarization")
        print("row 0:", np.mean(polu[:,0]))
        print("row 1:", np.mean(polu[:,1]))
        print("row 2:", np.mean(polu[:,2]))
        print("row 3:", np.mean(polu[:,3]))
        print("row 4:", np.mean(polu[:,4]))
        print("row 5:", np.mean(polu[:,5]))

        print("y polarization")
        print("row 0:", np.mean(polv[:,0]))
        print("row 1:", np.mean(polv[:,1]))
        print("row 2:", np.mean(polv[:,2]))
        print("row 3:", np.mean(polv[:,3]))
        print("row 4:", np.mean(polv[:,4]))
        print("row 5:", np.mean(polv[:,5]))

        ax1 = plt.subplot(141, ymargin=2)
        plt.title("(a) Polarization")
        Q = ax1.quiver(polu,polv, pivot="mid")
        ax1.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 C/m$^2$')
        plt.ylim(-0.4,7.4)
        plt.xlabel("x (unit cells)", fontsize=12)
        plt.ylabel("y (unit cells)", fontsize=12)

        x_dist, y_dist, z_dist = perovskite_FE_full(config, LABELS)

        feu = x_dist[:,:,0]
        fev = y_dist[:,:,0]

        ax2 = plt.subplot(142)
        plt.title("(b) FE mode")
        Q = ax2.quiver(feu,fev, pivot="mid")
        ax2.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 bohr')
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)
        
        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")
        
        afdau = x_rot[:,:,0]
        afdav = y_rot[:,:,0]

        ax3 = plt.subplot(143)
        Q = ax3.quiver(afdau,afdav, pivot="mid")
        plt.title(r"(c) AFD$^a$ mode")
        ax3.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="i")
        
        afdiu = x_rot[:,:,0]
        afdiv = y_rot[:,:,0]

        ax4 = plt.subplot(144)
        plt.title(r"(c) AFD$^i$ mode")
        Q = ax4.quiver(afdiu,afdiv, pivot="mid", scale=6, scale_units="xy")
        ax4.quiverkey(Q, 0.85, 1.03, 3, r'3º')
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        print("Gráfica finalizada: expansivo_dominios0.png")
        plt.savefig("FINAL_PLOTS/expansivo_dominios0.png")
        plt.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dominios expansivo plano xy T=50K ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_expansivo_1:

        config = sim.access(TEMPERATURES[1], s=STRAINS[0])

        plt.figure("expansivo_dominios1", figsize=[16,5])
        plt.subplots_adjust(left=0.03, bottom=0.12, right=0.99, top=0.9, wspace=0.02, hspace=0)
        plt.tight_layout()

        plane = 0

        polx, poly, polz = perovskite_polarization(config, LABELS, BORN_CHARGES)

        polu = polx[:,:,plane]
        polv = poly[:,:,plane]

        ax1 = plt.subplot(141, ymargin=2)
        plt.title("(a) Polarization")
        Q = ax1.quiver(polu,polv, pivot="mid")
        ax1.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 C/m$^2$')
        plt.ylim(-0.4,7.4)
        plt.xlabel("x (unit cells)", fontsize=12)
        plt.ylabel("y (unit cells)", fontsize=12)

        x_dist, y_dist, z_dist = perovskite_FE_full(config, LABELS)

        feu = x_dist[:,:,plane]
        fev = y_dist[:,:,plane]

        ax2 = plt.subplot(142)
        plt.title("(b) FE mode")
        Q = ax2.quiver(feu,fev, pivot="mid")
        ax2.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 bohr')
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)
        
        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")
        #x_rot, y_rot, z_rot = perovskite_simple_rotation(config, LABELS)
        
        afdau = x_rot[:,:,plane]
        afdav = y_rot[:,:,plane]

        ax3 = plt.subplot(143)
        Q = ax3.quiver(afdau,afdav, pivot="mid")
        plt.title(r"(c) AFD$^a$ mode")
        ax3.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="i")
        
        afdiu = x_rot[:,:,plane]
        afdiv = y_rot[:,:,plane]

        ax4 = plt.subplot(144)
        plt.title(r"(d) AFD$^i$ mode")
        Q = ax4.quiver(afdiu,afdiv, pivot="mid", scale=6, scale_units="xy")
        ax4.quiverkey(Q, 0.85, 1.03, 3, r'3º')
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        print("Gráfica finalizada: expansivo_dominios1.png")
        plt.savefig("FINAL_PLOTS/expansivo_dominios1.png")
        plt.draw()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_expansivo_1_extra:

        config = sim.access(TEMPERATURES[1], s=STRAINS[0])

        plt.figure("expansivo_dominios1_extra", figsize=[5,10])
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.97, top=0.95, wspace=0.02, hspace=0.1)
        plt.tight_layout()

        plane = 0
        
        x_rot, y_rot, z_rot = perovskite_simple_rotation(config, LABELS)
        
        afdau = x_rot[:,:,plane]
        afdav = y_rot[:,:,plane]

        ax1 = plt.subplot(211)
        Q = ax1.quiver(afdau,afdav, pivot="mid")
        plt.title(r"(a) Simple cell rotations")
        ax1.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax1.set_xticklabels([])
        #ax1.set_xticks([])
        plt.ylabel("y (unit cells)", fontsize=12)

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")
        #x_rot, y_rot, z_rot = perovskite_simple_rotation(config, LABELS)
        
        afdau = x_rot[:,:,plane]
        afdav = y_rot[:,:,plane]

        ax2 = plt.subplot(212)
        Q = ax2.quiver(afdau,afdav, pivot="mid")
        plt.title(r"(b) AFD$^a$ mode")
        ax2.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        
        plt.ylabel("y (unit cells)", fontsize=12)
        plt.xlabel("x (unit cells)", fontsize=12)



        print("Gráfica finalizada: expansivo_dominios1_extra.png")
        plt.savefig("FINAL_PLOTS/expansivo_dominios1_extra.png")
        plt.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dominios expansivo plano xy T=80K ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_expansivo_2:

        config = sim.access(TEMPERATURES[2], s=STRAINS[0])

        plt.figure("expansivo_dominios2", figsize=[16,5])
        plt.subplots_adjust(left=0.03, bottom=0.12, right=0.99, top=0.9, wspace=0.02, hspace=0)
        plt.tight_layout()

        polx, poly, polz = perovskite_polarization(config, LABELS, BORN_CHARGES)

        polu = polx[:,:,0]
        polv = poly[:,:,0]

        ax1 = plt.subplot(141, ymargin=2)
        plt.title("(a) Polarization")
        Q = ax1.quiver(polu,polv, pivot="mid")
        ax1.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 C/m$^2$')
        plt.ylim(-0.4,7.4)
        plt.xlabel("x (unit cells)", fontsize=12)
        plt.ylabel("y (unit cells)", fontsize=12)

        x_dist, y_dist, z_dist = perovskite_FE_full(config, LABELS)

        feu = x_dist[:,:,0]
        fev = y_dist[:,:,0]

        ax2 = plt.subplot(142)
        plt.title("(b) FE mode")
        Q = ax2.quiver(feu,fev, pivot="mid")
        ax2.quiverkey(Q, 0.85, 1.03, 0.4, r'0.4 bohr')
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)
        
        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")
        
        afdau = x_rot[:,:,0]
        afdav = y_rot[:,:,0]

        ax3 = plt.subplot(143)
        Q = ax3.quiver(afdau,afdav, pivot="mid")
        plt.title(r"(c) AFD$^a$ mode")
        ax3.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="i")
        
        afdiu = x_rot[:,:,0]
        afdiv = y_rot[:,:,0]

        ax4 = plt.subplot(144)
        plt.title(r"(d) AFD$^i$ mode")
        Q = ax4.quiver(afdiu,afdiv, pivot="mid", scale=6, scale_units="xy")
        ax4.quiverkey(Q, 0.85, 1.03, 3, r'3º')
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        print("Gráfica finalizada: expansivo_dominios2.png")
        plt.savefig("FINAL_PLOTS/expansivo_dominios2.png")
        plt.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dominios rotaciones compresivo xz yz ~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_expansivo_7:

        config = sim.access(TEMPERATURES[6], s=STRAINS[0])

        plt.figure("compresivo_dominios7", figsize=[5,5])
        plt.subplots_adjust(left=0.05, bottom=0.12, right=0.95, top=0.9, wspace=0.02, hspace=0)
        plt.tight_layout()

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")

        u = x_rot[:,:,0]
        v = y_rot[:,:,0]

        plt.title(r"(a) AFD$^a$ x-y plane")
        Q = plt.quiver(u,v, pivot="mid")
        plt.quiverkey(Q, 0.85, 1.03, 5, r'5 º')
        plt.ylim(-0.4,7.4)
        plt.xlabel("x (unit cells)", fontsize=12)
        plt.ylabel("y (unit cells)", fontsize=12)

        print("Gráfica finalizada: expansivo_dominios7.png")
        plt.savefig("FINAL_PLOTS/expansivo_dominios7.png")
        plt.draw()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dominios rotaciones compresivo xz yz ~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_compresivo_1:

        config = sim.access(TEMPERATURES[1], s=STRAINS[2])

        plt.figure("compresivo_dominios1", figsize=[16,5])
        plt.subplots_adjust(left=0.03, bottom=0.12, right=0.99, top=0.9, wspace=0.02, hspace=0)
        plt.tight_layout()

        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")

        u = x_rot[:,0,:]
        v = z_rot[:,0,:]

        u = np.rot90(u, k=3)
        v = np.rot90(v, k=3)


        ax1 = plt.subplot(141, ymargin=2)
        plt.title(r"(a) AFD$^a$ x-z plane")
        Q = ax1.quiver(u,v, pivot="mid")
        ax1.quiverkey(Q, 0.85, 1.03, 5, r'5 º')
        plt.ylim(-0.4,7.4)
        plt.xlabel("x (unit cells)", fontsize=12)
        plt.ylabel("z (unit cells)", fontsize=12)

        #x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="a")
        
        u = y_rot[0,:,:]
        v = z_rot[0,:,:]

        u = np.rot90(u, k=3)
        v = np.rot90(v, k=3)

        ax2 = plt.subplot(142)
        plt.title(r"(b) AFD$^a$ y-z plane")
        Q = ax2.quiver(u,v, pivot="mid")
        ax2.quiverkey(Q, 0.85, 1.03, 5, r"5º")
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.xlabel("y (unit cells)", fontsize=12)
        
        x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="i")
        
        u = x_rot[:,0,:]
        v = z_rot[:,0,:]

        u = np.rot90(u, k=3)
        v = np.rot90(v, k=3)

        ax3 = plt.subplot(143)
        Q = ax3.quiver(u,v, pivot="mid", scale=12, scale_units="xy")
        plt.title(r"(c) AFD$^i$ x-z plane")
        ax3.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        plt.xlabel("x (unit cells)", fontsize=12)

        #x_rot, y_rot, z_rot = perovskite_AFD(config, LABELS, mode="i")
        
        u = y_rot[0,:,:]
        v = z_rot[0,:,:]

        u = np.rot90(u, k=3)
        v = np.rot90(v, k=3)

        ax4 = plt.subplot(144)
        plt.title(r"(d) AFD$^i$ x-y plane")
        Q = ax4.quiver(u,v, pivot="mid", scale=12, scale_units="xy")
        ax4.quiverkey(Q, 0.85, 1.03, 5, r'5º')
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        plt.xlabel("y (unit cells)", fontsize=12)

        print("Gráfica finalizada: compresivo_dominios1.png")
        plt.savefig("FINAL_PLOTS/compresivo_dominios1.png")
        plt.draw()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3D rotation vector field compressive ~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if plot_3Dcompresivo_1:

        config = sim.access(TEMPERATURES[1], s=STRAINS[2])

        fig = plt.figure("compresivo_3Ddominios1", figsize=[8,8])
        ax = fig.gca(projection='3d')

        x, y, z = np.meshgrid(
            np.linspace(0,int(SUPERCELL[0])-1, SUPERCELL[0]),
            np.linspace(0,int(SUPERCELL[1])-1, SUPERCELL[1]),
            np.linspace(0,int(SUPERCELL[2])-1, SUPERCELL[2]),
        )
        
        u, v ,w = perovskite_AFD(config, LABELS, mode="a")
        ax.quiver(x, y, z, u, v, w, length=0.05, lw=2, color="blue", pivot = "middle")

        u, v ,w = perovskite_AFD(config, LABELS, mode="i")
        ax.quiver(x, y, z, u, v, w, length=0.07, lw=2, color="black", pivot = "middle")
        

        ax.set_xlabel("x (unit cells)", fontsize=12)
        ax.set_ylabel("y (unit cells)", fontsize=12)
        ax.set_zlabel("z (unit cells)", fontsize=12)

        
        ax.w_xaxis.set_pane_color((1,1,1,0))
        ax.w_yaxis.set_pane_color((1,1,1,0))
        ax.w_zaxis.set_pane_color((1,1,1,0))

        ax.w_xaxis.line.set_color((1,1,1,0))
        ax.w_yaxis.line.set_color((1,1,1,0))
        ax.w_zaxis.line.set_color((1,1,1,0))

        ax.set_xticks(np.linspace(0,int(SUPERCELL[0])-1, SUPERCELL[0]))
        ax.set_yticks(np.linspace(0,int(SUPERCELL[1])-1, SUPERCELL[1]))
        ax.set_zticks(np.linspace(0,int(SUPERCELL[2])-1, SUPERCELL[2]))
        

        plt.tight_layout(pad = 2)
        plt.grid(False)

        print("Gráfica finalizada: compresivo_3Ddominios1.png")
        plt.savefig("FINAL_PLOTS/compresivo_3Ddominios1.png")
        plt.draw()

    #config = sim.access(TEMPERATURES[1], s=STRAINS[0])
    #config.write_eq_geometry("eq.restart")

   
    if show_plots:
        print("\n Displaying plots...")
        plt.show()

    
    print("\nEVERYTHING DONE!")

#####################################################################