"""
Class created to analyze the output of MC simulations.
"""

# third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# standard library imports
from copy import deepcopy   # proper array copy
from pathlib import Path
import os, sys

# package imports
from ezSCUP.perovskite.modes import perovskite_AFD, perovskite_FE
from ezSCUP.perovskite.modes import perovskite_polarization
from ezSCUP.montecarlo import MCSimulationParser
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class PKAnalyzer()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class PKAnalyzer(MCSimulationParser):

    def __init__(self, labels, born_charges,
        output_folder="output"):
        
        super().__init__(output_folder=output_folder)

        self.labels = labels
        self.born_charges = born_charges
        self.fplots = os.path.join(output_folder, "_PLOTS")
        self.ffiles = os.path.join(output_folder, "_DATA")

        # plot settings
        self.label_size = 14
        self.figure_pad = 2

        # create analysis folders
        try: #create the plots folder if needed
            os.mkdir(self.fplots)
        except FileExistsError:
            pass
        try: #create the files folder if needed
            os.mkdir(self.ffiles)
        except FileExistsError:
            pass

        for i,_ in enumerate(self.strain):
            for j,_ in enumerate(self.stress):
                for k,_ in enumerate(self.field):

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    try:
                        os.mkdir(fplot)
                    except FileExistsError:
                        pass

                    try:
                        os.mkdir(ffile)
                    except FileExistsError:
                        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def AFD_vs_T(self, mode="a", rotate=False, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    rots = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)

                        angles = perovskite_AFD(geom, self.labels, mode=mode, angles=True)

                        if abs:
                            xrot, xrot_err = np.mean(np.abs(angles[:,:,:,0])), np.std(np.abs(angles[:,:,:,0]))
                            yrot, yrot_err = np.mean(np.abs(angles[:,:,:,1])), np.std(np.abs(angles[:,:,:,1]))
                            zrot, zrot_err = np.mean(np.abs(angles[:,:,:,2])), np.std(np.abs(angles[:,:,:,2]))
                        else:
                            xrot, xrot_err = np.mean(angles[:,:,:,0]), np.std(angles[:,:,:,0])
                            yrot, yrot_err = np.mean(angles[:,:,:,1]), np.std(angles[:,:,:,1])
                            zrot, zrot_err = np.mean(angles[:,:,:,2]), np.std(angles[:,:,:,2])

                        # rotate to make the biggest rotation always z axis                        
                        if rotate:
                            aux = np.abs(np.array([xrot, yrot, zrot]))
                            main_axis = np.argmax(aux)
                            if main_axis == 2:
                                rots[l,0,:] = [xrot, yrot, zrot]
                                rots[l,1,:] = [xrot_err, yrot_err, zrot_err]
                            if main_axis == 1:
                                rots[l,0,:] = [zrot, xrot, yrot]
                                rots[l,1,:] = [zrot_err, xrot_err, yrot_err]
                            else:
                                rots[l,0,:] = [yrot, zrot, xrot]
                                rots[l,1,:] = [yrot_err, zrot_err, xrot_err]
                        else:
                            rots[l,0,:] = [xrot, yrot, zrot]
                            rots[l,1,:] = [xrot_err, yrot_err, zrot_err]
                        pass

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    if abs:
                        plt.figure("AFD" + mode + "_abs.png")
                    else:
                        plt.figure("AFD" + mode + ".png")

                    plt.errorbar(self.temp, np.abs(rots[:,0,0]), yerr=rots[:,1,0], label=r"AFD$_{x}^{" + mode + "}$", marker ="<") 
                    plt.errorbar(self.temp, np.abs(rots[:,0,1]), yerr=rots[:,1,1], label=r"AFD$_{y}^{" + mode + "}$", marker =">") 
                    plt.errorbar(self.temp, np.abs(rots[:,0,2]), yerr=rots[:,1,2], label=r"AFD$_{z}^{" + mode + "}$", marker ="^") 
                    plt.ylabel("AFD$^{" + mode + "}$ (deg)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    plt.ylim(0, 10)
                    plt.grid(True)
                    
                    if abs:
                        plt.savefig(os.path.join(fplot, "AFD" + mode + "_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "AFD" + mode + ".png"))

                    index = 1
                    data = np.zeros((len(self.temp), 7))
                    data[:,0] = self.temp
                    for dim in range(3):
                        for value in range(2): 
                            data[:,index] = rots[:,value, dim]
                            index += 1
                    headers = ["temp", "xrot", "xrot_err", "yrot", 
                    "yrot_err", "zrot", "zrot_err"]

                    df = pd.DataFrame(data, columns=headers)
                    if abs:
                        df.to_csv(os.path.join(ffile, "AFD" + mode + "_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "AFD" + mode + ".csv"), index=False)
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def FE_vs_T(self, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    dists = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)
                        disps = perovskite_FE(geom, self.labels)

                        if abs:
                            xdist, xdist_err = np.mean(np.abs(disps[:,:,:,0])), np.std(np.abs(disps[:,:,:,0]))
                            ydist, ydist_err = np.mean(np.abs(disps[:,:,:,1])), np.std(np.abs(disps[:,:,:,1]))
                            zdist, zdist_err = np.mean(np.abs(disps[:,:,:,2])), np.std(np.abs(disps[:,:,:,2]))
                        else:
                            xdist, xdist_err = np.mean(disps[:,:,:,0]), np.std(disps[:,:,:,0])
                            ydist, ydist_err = np.mean(disps[:,:,:,1]), np.std(disps[:,:,:,1])
                            zdist, zdist_err = np.mean(disps[:,:,:,2]), np.std(disps[:,:,:,2])

                        dists[l,0,:] = [xdist, ydist, zdist]
                        dists[l,1,:] = [xdist_err, ydist_err, zdist_err]
                        pass

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("FE.png")
                    plt.errorbar(self.temp, dists[:,0,0], yerr=dists[:,1,0], label=r"FE$_{x}$", marker ="<") 
                    plt.errorbar(self.temp, dists[:,0,1], yerr=dists[:,1,1], label=r"FE$_{y}$", marker =">") 
                    plt.errorbar(self.temp, dists[:,0,2], yerr=dists[:,1,2], label=r"FE$_{z}$", marker ="^") 
                    plt.ylabel("FE (bohr)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    
                    if abs:
                        plt.savefig(os.path.join(fplot, "FEvsT_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "FEvsT.png"))

                    index = 1
                    data = np.zeros((len(self.temp), 7))
                    data[:,0] = self.temp
                    for dim in range(3):
                        for value in range(2): 
                            data[:,index] = dists[:,value, dim]
                            index += 1
                    headers = ["temp", "xdist", "xdist_err", "ydist", 
                    "ydist_err", "zdist", "zdist_err"]

                    df = pd.DataFrame(data, columns=headers)
                    if abs:
                        df.to_csv(os.path.join(ffile, "FEvsT_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "FEvsT.csv"), index=False)
        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def STRA_vs_T(self):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    strains = np.zeros((len(self.temp), 2, 6))
                    for l, t in enumerate(self.temp):

                        data = self.access_lattice_output(t, s=s, p=p, f=f)
                        data = data[data.index > self.mc_equilibration_steps] 

                        xxstra, xxstra_err = data["Strn_xx"].mean(), data["Strn_xx"].std()
                        yystra, yystra_err = data["Strn_yy"].mean(), data["Strn_yy"].std()
                        zzstra, zzstra_err = data["Strn_zz"].mean(), data["Strn_zz"].std()
                        yzstra, yzstra_err = data["Strn_yz"].mean(), data["Strn_yz"].std()
                        xzstra, xzstra_err = data["Strn_xz"].mean(), data["Strn_xz"].std()
                        xystra, xystra_err = data["Strn_xy"].mean(), data["Strn_xy"].std()

                        strains[l,0,:] = [xxstra, yystra, zzstra, yzstra, xzstra, xystra]
                        strains[l,1,:] = [xxstra_err, yystra_err, zzstra_err, 
                                        yzstra_err, xzstra_err, xystra_err]
                        pass

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("STRAvsT.png")
                    plt.errorbar(self.temp, strains[:,0,0]*100, yerr=strains[:,1,0]*100, label=r"$\eta_{xx}$", marker ="<") 
                    plt.errorbar(self.temp, strains[:,0,1]*100, yerr=strains[:,1,1]*100, label=r"$\eta_{yy}$", marker =">") 
                    plt.errorbar(self.temp, strains[:,0,2]*100, yerr=strains[:,1,2]*100, label=r"$\eta_{zz}$", marker ="^") 
                    plt.ylabel(r"$\eta$ (%)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    
                    plt.savefig(os.path.join(fplot, "STRAvsT.png"))
                    
                    index = 1
                    data = np.zeros((len(self.temp), 13))
                    data[:,0] = self.temp
                    for dim in range(6):
                        for value in range(2): 
                            data[:,index] = strains[:,value,dim]
                            index += 1
                    headers = ["temp", "sxx", "sxx_err", "syy", "syy_err",
                    "szz", "szz_err", "syz", "syz_err", "sxz", "sxz_err",
                    "sxy", "sxy_err"]

                    df = pd.DataFrame(data, columns=headers)
                    df.to_csv(os.path.join(ffile, "STRAvsT.csv"), index=False)
        pass

    def POL_vs_T(self, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    pols = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)
                        disps = perovskite_polarization(geom, self.labels, self.born_charges)

                        if abs:
                            xdist, xdist_err = np.mean(np.abs(disps[:,:,:,0])), np.std(np.abs(disps[:,:,:,0]))
                            ydist, ydist_err = np.mean(np.abs(disps[:,:,:,1])), np.std(np.abs(disps[:,:,:,1]))
                            zdist, zdist_err = np.mean(np.abs(disps[:,:,:,2])), np.std(np.abs(disps[:,:,:,2]))
                        else:
                            xdist, xdist_err = np.mean(disps[:,:,:,0]), np.std(disps[:,:,:,0])
                            ydist, ydist_err = np.mean(disps[:,:,:,1]), np.std(disps[:,:,:,1])
                            zdist, zdist_err = np.mean(disps[:,:,:,2]), np.std(disps[:,:,:,2])

                        pols[l,0,:] = [xdist, ydist, zdist]
                        pols[l,1,:] = [xdist_err, ydist_err, zdist_err]
                        pass

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("POLvsT.png")
                    plt.errorbar(self.temp, pols[:,0,0], yerr=pols[:,1,0], label=r"$P_{x}$", marker ="<") 
                    plt.errorbar(self.temp, pols[:,0,1], yerr=pols[:,1,1], label=r"$P_{y}$", marker =">") 
                    plt.errorbar(self.temp, pols[:,0,2], yerr=pols[:,1,2], label=r"$P_{z}$", marker ="^") 
                    plt.ylabel("$P$ (C/m)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    
                    if abs:
                        plt.savefig(os.path.join(fplot, "POLvsT_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "POLvsT.png"))

                    index = 1
                    data = np.zeros((len(self.temp), 7))
                    data[:,0] = self.temp
                    for dim in range(3):
                        for value in range(2): 
                            data[:,index] = pols[:,value, dim]
                            index += 1
                    headers = ["temp", "xpol", "xpol_err", "ypol", 
                    "ypol_err", "zpol", "zpol_err"]

                    df = pd.DataFrame(data, columns=headers)
                    if abs:
                        df.to_csv(os.path.join(ffile, "POLvsT_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "POLvsT.csv"), index=False)
        pass

















