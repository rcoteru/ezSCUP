"""

Generates the graphs presented in the thesis 
from the simulation data created after running "main.py"

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

try: #create the "plots" folder if needed
    os.mkdir("FINAL_PLOTS")
except FileExistsError:
    pass

plot_expansivo = True
plot_estatico = True
plot_compresivo = True

titlesize = 14
labelsize = 14
legendsize = 14

# CASO EXPANSIVO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if plot_expansivo:

    AFDa = pd.read_csv("csv/AFDa0.csv")
    AFDi = pd.read_csv("csv/AFDi0.csv")
    strain = pd.read_csv("csv/strain0.csv")

    FE = pd.read_csv("csv/fullFE0.csv")
    absFE = pd.read_csv("csv/abs_fullFE0.csv")
    pol = pd.read_csv("csv/polarization0.csv")
    abspol = pd.read_csv("csv/abs_polarization0.csv")

    # corrección
    AFDa.yrot, AFDa.xrot = np.where(AFDa.xrot > AFDa.yrot, [AFDa.yrot, AFDa.xrot], [AFDa.xrot, AFDa.yrot])

    # grafica AFD + strain
    plt.figure("expansivo_AFDstra", figsize=[16,5])
    plt.subplots_adjust(left=0.03, bottom=0.12, right=0.96, top=0.9, wspace=0.02, hspace=0)

    ax1 = plt.subplot(131)
    plt.title("(a) AFD$^{a}$ mode", fontsize=titlesize)
    plt.errorbar(AFDa["temp"], AFDa["xrot"], yerr=AFDa["xrot_err"], label=r"AFD$_{x}^{a}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDa["temp"], AFDa["yrot"], yerr=AFDa["yrot_err"], label=r"AFD$_{y}^{a}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDa["temp"], AFDa["zrot"], yerr=AFDa["zrot_err"], label=r"AFD$_{z}^{a}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylabel(r"AFD rotation (deg)", fontsize = labelsize)
    plt.ylim(0,8)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax2 = plt.subplot(132)
    ax2.set_yticklabels([])
    plt.title("(b) AFD$^{i}$ mode", fontsize=titlesize)
    plt.errorbar(AFDi["temp"], AFDi["xrot"], yerr=AFDi["xrot_err"], label=r"AFD$_{x}^{i}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDi["temp"], AFDi["yrot"], yerr=AFDi["yrot_err"], label=r"AFD$_{y}^{i}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDi["temp"], AFDi["zrot"], yerr=AFDi["zrot_err"], label=r"AFD$_{z}^{i}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylim(0,8)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax3 = plt.subplot(133)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    plt.title("(c) Strain", fontsize=titlesize)
    plt.errorbar(strain["temp"], strain["sx"]*100, yerr=strain["sx_err"]*100, label=r"$\eta_x$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(strain["temp"], strain["sy"]*100, yerr=strain["sy_err"]*100, label=r"$\eta_y$", marker =">", linestyle="-", c="black") 
    plt.errorbar(strain["temp"], strain["sz"]*100, yerr=strain["sz_err"]*100, label=r"$\eta_z$", marker ="^", linestyle="-", c="red")
    plt.ylabel(r"$\eta$ (%)", fontsize = labelsize, labelpad=1)
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    plt.savefig("FINAL_PLOTS/expansivo_AFDstra.png")
    plt.draw()

    # grafica FE + pol
    plt.figure("expansivo_FEpol", figsize=[12,5])
    plt.subplots_adjust(left=0.06, bottom=0.12, right=0.93, top=0.9, wspace=0.02, hspace=0)

    ax1 = plt.subplot(121)
    plt.title("(a) FE mode", fontsize=titlesize)
    plt.errorbar(absFE["temp"], absFE["xdist"], yerr=absFE["xdist_err"], label=r"FE$_{x}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(absFE["temp"], absFE["ydist"], yerr=absFE["ydist_err"], label=r"FE$_{y}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(absFE["temp"], absFE["zdisr"], yerr=absFE["zdist_err"], label=r"FE$_{z}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylim(0, 0.35)
    plt.ylabel(r"FE displacement (bohr)", fontsize = labelsize)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)


    ax3 = plt.subplot(122)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    plt.title("(b) Polarization", fontsize=titlesize)
    plt.errorbar(abspol["temp"], abspol["px"], yerr=abspol["px_err"], label=r"P$_{x}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(abspol["temp"], abspol["py"], yerr=abspol["py_err"], label=r"P$_{y}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(abspol["temp"], abspol["pz"], yerr=abspol["pz_err"], label=r"P$_{z}$", marker ="^", linestyle="-", c="red")
    plt.ylabel("$P$ (C/m$^2$)", fontsize = labelsize)
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylim(0, 0.25)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)


    plt.savefig("FINAL_PLOTS/expansivo_FEpol.png")
    plt.draw()

# CASO ESTÁTICO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if plot_estatico:

    AFDa = pd.read_csv("csv/AFDa1.csv")
    AFDi = pd.read_csv("csv/AFDi1.csv")
    strain = pd.read_csv("csv/strain1.csv")

    FE = pd.read_csv("csv/fullFE1.csv")
    absFE = pd.read_csv("csv/abs_fullFE1.csv")
    pol = pd.read_csv("csv/polarization1.csv")
    abspol = pd.read_csv("csv/abs_polarization1.csv")

    # corrección
    AFDa.yrot, AFDa.xrot = np.where(AFDa.xrot > AFDa.yrot, [AFDa.yrot, AFDa.xrot], [AFDa.xrot, AFDa.yrot])

    # grafica AFD + strain
    plt.figure("estatico_AFDstra", figsize=[16,5])
    plt.subplots_adjust(left=0.03, bottom=0.12, right=0.95, top=0.9, wspace=0.02, hspace=0)

    ax1 = plt.subplot(131)
    plt.title("(a) AFD$^{a}$ mode", fontsize=titlesize)
    plt.errorbar(AFDa["temp"], AFDa["xrot"], yerr=AFDa["xrot_err"], label=r"AFD$_{x}^{a}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDa["temp"], AFDa["yrot"], yerr=AFDa["yrot_err"], label=r"AFD$_{y}^{a}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDa["temp"], AFDa["zrot"], yerr=AFDa["zrot_err"], label=r"AFD$_{z}^{a}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylabel(r"AFD rotation (deg)", fontsize = labelsize)
    plt.ylim(0,6)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax2 = plt.subplot(132)
    ax2.set_yticklabels([])
    plt.title("(b) AFD$^{i}$ mode", fontsize=titlesize)
    plt.errorbar(AFDi["temp"], AFDi["xrot"], yerr=AFDi["xrot_err"], label=r"AFD$_{x}^{i}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDi["temp"], AFDi["yrot"], yerr=AFDi["yrot_err"], label=r"AFD$_{y}^{i}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDi["temp"], AFDi["zrot"], yerr=AFDi["zrot_err"], label=r"AFD$_{z}^{i}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylim(0,6)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax3 = plt.subplot(133)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    plt.title("(c) Strain", fontsize=titlesize)
    plt.errorbar(strain["temp"], strain["sx"]*100, yerr=strain["sx_err"]*100, label=r"$\eta_x$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(strain["temp"], strain["sy"]*100, yerr=strain["sy_err"]*100, label=r"$\eta_y$", marker =">", linestyle="-", c="black") 
    plt.errorbar(strain["temp"], strain["sz"]*100, yerr=strain["sz_err"]*100, label=r"$\eta_z$", marker ="^", linestyle="-", c="red")
    plt.ylabel(r"$\eta$ (%)", fontsize = labelsize)
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    plt.savefig("FINAL_PLOTS/estatico_AFDstra.png")
    plt.draw()


# CASO COMPRESIVO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if plot_compresivo:

    AFDa = pd.read_csv("csv/AFDa2.csv")
    AFDi = pd.read_csv("csv/AFDi2.csv")
    strain = pd.read_csv("csv/strain2.csv")

    FE = pd.read_csv("csv/fullFE2.csv")
    absFE = pd.read_csv("csv/abs_fullFE2.csv")
    pol = pd.read_csv("csv/polarization2.csv")
    abspol = pd.read_csv("csv/abs_polarization2.csv")

    # corrección
    AFDa.yrot, AFDa.xrot = np.where(AFDa.xrot > AFDa.yrot, [AFDa.yrot, AFDa.xrot], [AFDa.xrot, AFDa.yrot])

    # grafica AFD + strain
    plt.figure("compresivo_AFDstra", figsize=[16,5])
    plt.subplots_adjust(left=0.03, bottom=0.12, right=0.96, top=0.9, wspace=0.02, hspace=0)

    ax1 = plt.subplot(131)
    plt.title("(a) AFD$^{a}$ mode", fontsize=titlesize)
    plt.errorbar(AFDa["temp"], AFDa["xrot"], yerr=AFDa["xrot_err"], label=r"AFD$_{x}^{a}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDa["temp"], AFDa["yrot"], yerr=AFDa["yrot_err"], label=r"AFD$_{y}^{a}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDa["temp"], AFDa["zrot"], yerr=AFDa["zrot_err"], label=r"AFD$_{z}^{a}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylabel(r"AFD rotation (deg)", fontsize = labelsize)
    plt.ylim(0,10)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax2 = plt.subplot(132)
    ax2.set_yticklabels([])
    plt.title("(b) AFD$^{i}$ mode", fontsize=titlesize)
    plt.errorbar(AFDi["temp"], AFDi["xrot"], yerr=AFDi["xrot_err"], label=r"AFD$_{x}^{i}$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(AFDi["temp"], AFDi["yrot"], yerr=AFDi["yrot_err"], label=r"AFD$_{y}^{i}$", marker =">", linestyle="-", c="black") 
    plt.errorbar(AFDi["temp"], AFDi["zrot"], yerr=AFDi["zrot_err"], label=r"AFD$_{z}^{i}$", marker ="^", linestyle="-", c="red")
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.ylim(0,10)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    ax3 = plt.subplot(133)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    plt.title("(c) Strain", fontsize=titlesize)
    plt.errorbar(strain["temp"], strain["sx"]*100, yerr=strain["sx_err"]*100, label=r"$\eta_x$", marker ="<", linestyle="-", c="blue") 
    plt.errorbar(strain["temp"], strain["sy"]*100, yerr=strain["sy_err"]*100, label=r"$\eta_y$", marker =">", linestyle="-", c="black") 
    plt.errorbar(strain["temp"], strain["sz"]*100, yerr=strain["sz_err"]*100, label=r"$\eta_z$", marker ="^", linestyle="-", c="red")
    plt.ylabel(r"$\eta$ (%)", fontsize = labelsize)
    plt.xlabel("$T$ (K)", fontsize = labelsize)
    plt.legend(frameon = True, fontsize = legendsize)
    plt.grid(True)

    plt.savefig("FINAL_PLOTS/compresivo_AFDstra.png")
    plt.draw()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if plot_expansivo or plot_estatico or plot_compresivo: 
    plt.show()
