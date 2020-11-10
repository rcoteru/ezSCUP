"""
Class created to analyze the output of MC simulations.
"""

# third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# standard library imports
import os, sys, shutil
import time

# package imports
from ezSCUP.perovskite.modes import perovskite_AFD, perovskite_FE
from ezSCUP.perovskite.modes import perovskite_polarization
from ezSCUP.montecarlo import MCSimulationParser
from ezSCUP.plotting import plot_vector, plot_vectors
from ezSCUP.singlepoint import SPRun
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class PKAnalyzer()
# 
# + class Timer()
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
        self.colors = ["black", "blue", "red", "green", "yellow"]
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
                    
                    dirs = [
                            os.path.join(self.ffiles, label),
                            os.path.join(self.fplots, label),
                            os.path.join(self.fplots, label, "AFDa"),
                            os.path.join(self.fplots, label, "AFDi"),
                            os.path.join(self.fplots, label, "FE"),
                            os.path.join(self.fplots, label, "POL"),
                            os.path.join(self.fplots, label, "STRA"),                            
                            ]

                    for d in dirs:
                        try:
                            os.mkdir(d)
                        except FileExistsError:
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def clear_output(self, data=True, plots=True):

        for i,_ in enumerate(self.strain):
            for j,_ in enumerate(self.stress):
                for k,_ in enumerate(self.field):

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label)
                    ffile = os.path.join(self.ffiles, label)

                    if plots:
                        for filename in os.listdir(fplot):
                            file_path = os.path.join(fplot, filename)
                            try:
                                if os.path.isfile(file_path) or os.path.islink(file_path):
                                    os.unlink(file_path)
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path)
                            except Exception as e:
                                print('Failed to delete %s. Reason: %s' % (file_path, e))

                    if data:
                        for filename in os.listdir(ffile):
                            file_path = os.path.join(fplot, filename)
                            try:
                                if os.path.isfile(file_path) or os.path.islink(file_path):
                                    os.unlink(file_path)
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path)
                            except Exception as e:
                                print('Failed to delete %s. Reason: %s' % (file_path, e))
                    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def AFD_vs_T(self, mode="a", rotate=None, abs=False):

        # TODO rotación plano xy

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
       
                        if rotate == "z": # rotate to make the biggest rotation always z axis      
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
                        if rotate == "xy": # rotate in-plane to make the x axis always the largest
                            aux = np.abs(np.array([xrot, yrot]))
                            main_axis = np.argmax(aux)
                            if main_axis == 0:
                                rots[l,0,:] = [xrot, yrot, zrot]
                                rots[l,1,:] = [xrot_err, yrot_err, zrot_err]
                            else: 
                                rots[l,0,:] = [yrot, xrot, zrot]
                                rots[l,1,:] = [yrot_err, xrot_err, zrot_err]
                        else:
                            rots[l,0,:] = [xrot, yrot, zrot]
                            rots[l,1,:] = [xrot_err, yrot_err, zrot_err]
                        pass

                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "AFD"+mode)
                    ffile = os.path.join(self.ffiles, label)

                    if abs:
                        plt.figure("AFD" + mode + "_abs.png")
                    else:
                        plt.figure("AFD" + mode + ".png")

                    plt.errorbar(self.temp, np.abs(rots[:,0,0]), yerr=rots[:,1,0], 
                    label=r"AFD$_{x}^{" + mode + "}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, np.abs(rots[:,0,1]), yerr=rots[:,1,1], 
                    label=r"AFD$_{y}^{" + mode + "}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, np.abs(rots[:,0,2]), yerr=rots[:,1,2], 
                    label=r"AFD$_{z}^{" + mode + "}$", marker ="^", c=self.colors[2]) 
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
                    plt.close()

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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
                    fplot = os.path.join(self.fplots, label, "FE")
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("FE.png")
                    plt.errorbar(self.temp, dists[:,0,0], yerr=dists[:,1,0], 
                    label=r"FE$_{x}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, dists[:,0,1], yerr=dists[:,1,1], 
                    label=r"FE$_{y}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, dists[:,0,2], yerr=dists[:,1,2], 
                    label=r"FE$_{z}$", marker ="^", c=self.colors[2]) 
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
                    plt.close()


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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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
                    fplot = os.path.join(self.fplots, label, "STRA")
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("STRAvsT.png")
                    plt.errorbar(self.temp, strains[:,0,0]*100, yerr=strains[:,1,0]*100, 
                    label=r"$\eta_{xx}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, strains[:,0,1]*100, yerr=strains[:,1,1]*100, 
                    label=r"$\eta_{yy}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, strains[:,0,2]*100, yerr=strains[:,1,2]*100, 
                    label=r"$\eta_{zz}$", marker ="^", c=self.colors[2]) 
                    plt.ylabel(r"$\eta$ (%)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    plt.savefig(os.path.join(fplot, "STRAvsT.png"))
                    plt.close()

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
 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

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
                    fplot = os.path.join(self.fplots, label, "POL")
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("POLvsT.png")
                    plt.errorbar(self.temp, pols[:,0,0], yerr=pols[:,1,0], 
                    label=r"$P_{x}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, pols[:,0,1], yerr=pols[:,1,1], 
                    label=r"$P_{y}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, pols[:,0,2], yerr=pols[:,1,2], 
                    label=r"$P_{z}$", marker ="^", c=self.colors[2]) 
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
                    plt.close()

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def ENERGY_vs_T(self):

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def AFD_horizontal_domain_vectors(self, layers, mode="a"):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "AFD"+mode)
                    
                    for _, t in enumerate(self.temp):

                        pname = "AFD" + mode + "dom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = perovskite_AFD(geom, self.labels, mode=mode, angles=True)

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize=12)
                            plt.ylabel("y (unit cells)", fontsize=12)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
                            
                            axs[0].quiver(u1,v1, pivot="mid")
                            axs[0].set_xlabel("x (unit cells)", fontsize=12)
                            axs[0].set_ylabel("y (unit cells)", fontsize=12)
                            axs[0].invert_yaxis()

                            axs[1].quiver(u2,v2, pivot="mid")
                            axs[1].set_xlabel("x (unit cells)", fontsize=12)
                            axs[1].invert_yaxis()

                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        else:
                            print("\nPLOTTING ERROR: Unsupported number of layers!")
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def FE_horizontal_domain_vectors(self, layers):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "FE")
                    
                    for _, t in enumerate(self.temp):

                        pname = "FEdom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = perovskite_FE(geom, self.labels)

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize=12)
                            plt.ylabel("y (unit cells)", fontsize=12)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
                            
                            axs[0].quiver(u1,v1, pivot="mid")
                            axs[0].set_xlabel("x (unit cells)", fontsize=12)
                            axs[0].set_ylabel("y (unit cells)", fontsize=12)
                            axs[0].invert_yaxis()

                            axs[1].quiver(u2,v2, pivot="mid")
                            axs[1].set_xlabel("x (unit cells)", fontsize=12)
                            axs[1].invert_yaxis()
                            
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        else:
                            print("\nPLOTTING ERROR: Unsupported number of layers!")
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def POL_horizontal_domain_vectors(self, layers):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "POL")
                    
                    for _, t in enumerate(self.temp):

                        pname = "POLdom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = perovskite_polarization(geom, self.labels, self.born_charges)

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize=12)
                            plt.ylabel("y (unit cells)", fontsize=12)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
                            
                            axs[0].quiver(u1,v1, pivot="mid")
                            axs[0].set_xlabel("x (unit cells)", fontsize=12)
                            axs[0].set_ylabel("y (unit cells)", fontsize=12)
                            axs[0].invert_yaxis()

                            axs[1].quiver(u2,v2, pivot="mid")
                            axs[1].set_xlabel("x (unit cells)", fontsize=12)
                            axs[1].invert_yaxis()
                            
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        else:
                            print("\nPLOTTING ERROR: Unsupported number of layers!")
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def POL_stepped(self):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "POL")
                    
                    for _, t in enumerate(self.temp):

                        pname = "POLstepped_" + str(int(t)) + ".png"

                        lat = self.access_lattice_output(t, s=s, p=p, f=f)

                        plt.figure(pname)
                        plt.plot(lat.index, lat["Pol_x(C/m2)"], label="$P_x$", c=self.colors[0])
                        plt.plot(lat.index, lat["Pol_y(C/m2)"], label="$P_y$", c=self.colors[1])
                        plt.plot(lat.index, lat["Pol_z(C/m2)"], label="$P_z$", c=self.colors[2])
                        plt.legend(frameon=True, fontsize = self.label_size)
                        plt.ylabel("$P$ (C/m2)", fontsize = self.label_size)
                        plt.xlabel("MC Step", fontsize = self.label_size)
                        plt.grid(True)
                        plt.savefig(os.path.join(fplot, pname))
                        plt.close()


        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #


class Timer:

    timers = dict()

    class TimerError(Exception):

        """A custom exception used to report errors in use of Timer class"""

    def __init__(
        self,
        name=None,
        text="Elapsed time: {:0.4f} seconds",
        logger=print,
    ):
        self._start_time = None
        self.name = name
        self.text = text
        self.logger = logger

        # Add new named timers to dictionary of timers
        if name:
            self.timers.setdefault(name, 0)

    def start(self):

        """Start a new timer"""

        if self._start_time is not None:
            raise self.TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):

        """Stop the timer, and report the elapsed time"""
        
        if self._start_time is None:
            raise self.TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time












