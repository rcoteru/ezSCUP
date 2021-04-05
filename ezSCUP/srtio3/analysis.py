"""
Class created to analyze the output of MC simulations.
"""

# third party imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

# standard library imports
import os, sys, shutil
import time

# package imports
from ezSCUP.srtio3.modes import STO_ROT, STO_AFD, STO_FE, STO_AFE, STO_OD, STO_POL
from ezSCUP.optimization import CGDSimulationParser
from ezSCUP.montecarlo import MCSimulationParser
from ezSCUP.singlepoint import SPRun
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class STO_MCAnalyzer()
#
# + class STO_CGDAnalyzer()
# 
# + class Timer()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

class STO_MCAnalyzer(MCSimulationParser):

    def __init__(self, output_folder="output"):

        #CHECK IF ITS AN STO SIMULATION
        
        super().__init__(output_folder=output_folder)

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

                    Tlabel = "s{:d}p{:d}f{:d}".format(i,j,k)
                    STRAlabel = "p{:d}f{:d}".format(j,k)

                    dirs = [
                            os.path.join(self.ffiles, Tlabel),
                            os.path.join(self.fplots, "T-dep"),
                            os.path.join(self.fplots, "T-dep", Tlabel),
                            os.path.join(self.fplots, "T-dep", Tlabel, "ROT"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "AFDa"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "AFDi"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "FE"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "AFE"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "OD"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "POL"),
                            os.path.join(self.fplots, "T-dep", Tlabel, "STRA"),                            
                            os.path.join(self.fplots, "STRA-dep"),
                            os.path.join(self.fplots, "STRA-dep", STRAlabel),
                            os.path.join(self.fplots, "STRA-dep", STRAlabel, "ROT")
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

    def ROT_vs_T(self, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    rots = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)

                        angles = STO_ROT(geom, self.model, angles=True)

                        if abs:
                            xrot, xrot_err = np.mean(np.abs(angles[:,:,:,0])), np.std(np.abs(angles[:,:,:,0]))
                            yrot, yrot_err = np.mean(np.abs(angles[:,:,:,1])), np.std(np.abs(angles[:,:,:,1]))
                            zrot, zrot_err = np.mean(np.abs(angles[:,:,:,2])), np.std(np.abs(angles[:,:,:,2]))
                        else:
                            xrot, xrot_err = np.mean(angles[:,:,:,0]), np.std(angles[:,:,:,0])
                            yrot, yrot_err = np.mean(angles[:,:,:,1]), np.std(angles[:,:,:,1])
                            zrot, zrot_err = np.mean(angles[:,:,:,2]), np.std(angles[:,:,:,2])
       
                        rots[l,0,:] = [xrot, yrot, zrot]
                        rots[l,1,:] = [xrot_err, yrot_err, zrot_err]


                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "T-dep", "ROT")
                    ffile = os.path.join(self.ffiles, label)

                    if abs:
                        plt.figure("ROT_abs.png")
                    else:
                        plt.figure("ROT.png")

                    plt.errorbar(self.temp, np.abs(rots[:,0,0]), yerr=rots[:,1,0], 
                    label=r"ROT$_{x}", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, np.abs(rots[:,0,1]), yerr=rots[:,1,1], 
                    label=r"ROT$_{y}", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, np.abs(rots[:,0,2]), yerr=rots[:,1,2], 
                    label=r"ROT$_{z}", marker ="^", c=self.colors[2]) 
                    plt.ylabel("ROT (deg)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    plt.ylim(0, 10)
                    plt.grid(True)
                    if abs:
                        plt.savefig(os.path.join(fplot, "ROT_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "ROT.png"))
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
                        df.to_csv(os.path.join(ffile, "ROT_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "ROT.csv"), index=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    def ROT_vs_STRA_fixed_T(self, abs=False):

        stra_list = np.array([s[0] for s in self.strain])

        for j,p in enumerate(self.stress):
            for k,f in enumerate(self.field):
                
                t = self.temp[0]

                rots = np.zeros((len(self.strain), 2, 3)) 
                for i, s in enumerate(self.strain):

                    geom = self.access_geometry(t, s=s, p=p, f=f)

                    angles = STO_ROT(geom, self.model, angles=True)

                    if abs:
                        xrot, xrot_err = np.mean(np.abs(angles[:,:,:,0])), np.std(np.abs(angles[:,:,:,0]))
                        yrot, yrot_err = np.mean(np.abs(angles[:,:,:,1])), np.std(np.abs(angles[:,:,:,1]))
                        zrot, zrot_err = np.mean(np.abs(angles[:,:,:,2])), np.std(np.abs(angles[:,:,:,2]))
                    else:
                        xrot, xrot_err = np.mean(angles[:,:,:,0]), np.std(angles[:,:,:,0])
                        yrot, yrot_err = np.mean(angles[:,:,:,1]), np.std(angles[:,:,:,1])
                        zrot, zrot_err = np.mean(angles[:,:,:,2]), np.std(angles[:,:,:,2])
    
                    rots[i,0,:] = [xrot, yrot, zrot]
                    rots[i,1,:] = [xrot_err, yrot_err, zrot_err]


                label = "p{:d}f{:d}".format(j,k)
                fplot = os.path.join(self.fplots, "STRA-dep", label, "ROT")
                ffile = os.path.join(self.ffiles, label)

                if abs:
                    plt.figure("ROTvsSTRA_abs.png")
                else:
                    plt.figure("ROTvsSTRA.png")

                plt.errorbar(stra_list*100, np.abs(rots[:,0,0]), yerr=rots[:,1,0], 
                label=r"ROT$_{x}", marker ="<", c=self.colors[0]) 
                plt.errorbar(stra_list*100, np.abs(rots[:,0,1]), yerr=rots[:,1,1], 
                label=r"ROT$_{y}", marker =">", c=self.colors[1]) 
                plt.errorbar(stra_list*100, np.abs(rots[:,0,2]), yerr=rots[:,1,2], 
                label=r"ROT$_{z}", marker ="^", c=self.colors[2]) 
                plt.ylabel("ROT (deg)", fontsize = self.label_size)
                plt.xlabel("Strain (%)", fontsize = self.label_size)
                plt.legend(frameon=True, fontsize = self.label_size)
                plt.tight_layout(pad = self.figure_pad)
                plt.ylim(0, 10)
                plt.grid(True)
                if abs:
                    plt.savefig(os.path.join(fplot, "ROTvsSTRA_abs.png"))
                else:
                    plt.savefig(os.path.join(fplot, "ROTvsSTRA.png"))
                plt.close()

               # index = 1
               # data = np.zeros((len(self.temp), 7))
               # data[:,0] = t
               # for dim in range(3):
               #     for value in range(2): 
               #         data[:,index] = rots[:,value, dim]
               #         index += 1
               # headers = ["temp", "xrot", "xrot_err", "yrot", 
               # "yrot_err", "zrot", "zrot_err"]

               # df = pd.DataFrame(data, columns=headers)
               # if abs:
               #     df.to_csv(os.path.join(ffile, "ROTvsSTRA_abs.csv"), index=False)
               # else:
               #     df.to_csv(os.path.join(ffile, "ROTvsSTRA.csv"), index=False)

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

                        angles = STO_AFD(geom, self.model, mode=mode, angles=True, algo="sign")

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
                        disps = STO_FE(geom, self.model)

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
                    fplot = os.path.join(self.fplots, label, "T-dep", "FE")
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

    def AFE_vs_T(self, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    dists = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)
                        disps = STO_AFE(geom, self.model)

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
                    fplot = os.path.join(self.fplots, label, "T-dep", "AFE")
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("AFE.png")
                    plt.errorbar(self.temp, dists[:,0,0], yerr=dists[:,1,0], 
                    label=r"AFE$_{x}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, dists[:,0,1], yerr=dists[:,1,1], 
                    label=r"AFE$_{y}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, dists[:,0,2], yerr=dists[:,1,2], 
                    label=r"AFE$_{z}$", marker ="^", c=self.colors[2]) 
                    plt.ylabel("AFE (bohr)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    if abs:
                        plt.savefig(os.path.join(fplot, "AFEvsT_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "AFEvsT.png"))
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
                        df.to_csv(os.path.join(ffile, "AFEvsT_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "AFEvsT.csv"), index=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def OD_vs_T(self, abs=False):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    dists = np.zeros((len(self.temp), 2, 3))
                    for l, t in enumerate(self.temp):

                        geom = self.access_geometry(t, s=s, p=p, f=f)
                        disps = STO_OD(geom, self.model)

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
                    fplot = os.path.join(self.fplots, label, "T-dep", "AFE")                    
                    ffile = os.path.join(self.ffiles, label)

                    plt.figure("OD.png")
                    plt.errorbar(self.temp, dists[:,0,0], yerr=dists[:,1,0], 
                    label=r"OD$_{x}$", marker ="<", c=self.colors[0]) 
                    plt.errorbar(self.temp, dists[:,0,1], yerr=dists[:,1,1], 
                    label=r"OD$_{y}$", marker =">", c=self.colors[1]) 
                    plt.errorbar(self.temp, dists[:,0,2], yerr=dists[:,1,2], 
                    label=r"OD$_{z}$", marker ="^", c=self.colors[2]) 
                    plt.ylabel("FE (bohr)", fontsize = self.label_size)
                    plt.xlabel("T (K)", fontsize = self.label_size)
                    plt.legend(frameon=True, fontsize = self.label_size)
                    plt.tight_layout(pad = self.figure_pad)
                    #plt.ylim(0, 10)
                    plt.grid(True)
                    if abs:
                        plt.savefig(os.path.join(fplot, "ODvsT_abs.png"))
                    else:
                        plt.savefig(os.path.join(fplot, "ODvsT.png"))
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
                        df.to_csv(os.path.join(ffile, "ODvsT_abs.csv"), index=False)
                    else:
                        df.to_csv(os.path.join(ffile, "ODvsT.csv"), index=False)

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
                    fplot = os.path.join(self.fplots, label, "T-dep", "STRA")                    
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
                        disps = STO_POL(geom, self.model)

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
                    fplot = os.path.join(self.fplots, label, "T-dep", "POL")                    
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

    def ROT_horizontal_domain_vectors(self, layers, symmetry="G"):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "T-dep", "ROT")
                    
                    for _, t in enumerate(self.temp):

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = STO_ROT(geom, self.model, angles=True)
                        
                        # setup symmetry
                        if symmetry == "G":
                            sym_fac = np.array([0,0,0])
                            pass
                        elif symmetry == "R":
                            sym_fac = np.array([1,1,1])
                            pass
                        else:
                            print("Unrecognized symmetry setting:" + str(symmetry))
                            print("Defaulting to G point.")
                            symmetry = "G"
                            sym_fac = np.array([0,0,0])

                        for x in range(geom.supercell[0]):
                            for y in range(geom.supercell[1]):
                                for z in range(geom.supercell[2]):
                                    cell = np.array([x,y,z])
                                    angles[x,y,z,:] = angles[x,y,z,:]*np.real(np.exp(-1j*np.pi*np.dot(cell, sym_fac)))

                        pname = "ROTdom" + symmetry + "sym_T" + str(int(t)) + ".png"
                        X, Y = np.meshgrid(range(geom.supercell[0]), range(geom.supercell[1]))

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(X, Y, u, v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize = self.label_size)
                            plt.ylabel("y (unit cells)", fontsize = self.label_size)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[13,7], sharey=True)
                            fig.suptitle("Octahedral Rotation @ $T={:d}$ K (sym={:s})".format(int(t), symmetry), fontsize = self.label_size)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
              
                            m = np.mean(np.hypot(u1, v1))
                            q0 = axs[0].quiver(X, Y, u1, v1, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[0].quiverkey(q0, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[0].set_title("$z={:d}$".format(layers[0]), fontsize = self.label_size)
                            axs[0].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[0].set_ylabel("y (unit cells)", fontsize = self.label_size)
                            axs[0].invert_yaxis()

                            m = np.mean(np.hypot(u2, v2))
                            q1 = axs[1].quiver(X, Y, u2, v2, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[1].quiverkey(q1, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[1].set_title("$z={:d}$".format(layers[1]), fontsize = self.label_size)
                            axs[1].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[1].invert_yaxis()

                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        else:
                            print("\nPLOTTING ERROR: Unsupported number of layers!")
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    def ROT_POL_horizontal_domain_vectors(self, layers, symmetry="G"):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "T-dep", "ROT")
                    
                    for _, t in enumerate(self.temp):

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = STO_ROT(geom, self.model, angles=True)
                        pols = STO_POL(geom, self.model)
                        
                        # setup symmetry for angles
                        if symmetry == "G":
                            sym_fac = np.array([0,0,0])
                            pass
                        elif symmetry == "R":
                            sym_fac = np.array([1,1,1])
                            pass
                        else:
                            print("Unrecognized symmetry setting:" + str(symmetry))
                            print("Defaulting to G point.")
                            symmetry = "G"
                            sym_fac = np.array([0,0,0])

                        # apply symmetry to rotations
                        for x in range(geom.supercell[0]):
                            for y in range(geom.supercell[1]):
                                for z in range(geom.supercell[2]):
                                    cell = np.array([x,y,z])
                                    angles[x,y,z,:] = angles[x,y,z,:]*np.real(np.exp(-1j*np.pi*np.dot(cell, sym_fac)))

                        pname = "ROT-POLdom" + symmetry + "sym_T" + str(int(t)) + ".png"
                        X, Y = np.meshgrid(range(geom.supercell[0]), range(geom.supercell[1]))
                        
                        shift = 0.1
                        X1, Y1 = X+shift, Y
                        X2, Y2 = X-shift, Y

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(X, Y, u, v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize = self.label_size)
                            plt.ylabel("y (unit cells)", fontsize = self.label_size)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            i1, j1 = pols[:,:,layers[0],0], pols[:,:,layers[0],1]
                            i2, j2 = pols[:,:,layers[1],0], pols[:,:,layers[1],1]

                            fig, axs = plt.subplots(1,2, figsize=[13,7], sharey=True)
                            fig.suptitle("Octahedral Rotation and Polarization @ $T={:d}$ K ({:s}-point Rotational Symmetry)".format(int(t), symmetry), fontsize = self.label_size)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
              
                            m = np.mean(np.hypot(u1, v1))
                            n = np.mean(np.hypot(i1, j1))
                            q0 = axs[0].quiver(X1, Y1, u1, v1, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            p0 = axs[0].quiver(X2, Y2, i1, j1, pivot="mid", width=0.008, headwidth=5, minlength=4, minshaft=3, color="r")
                            axs[0].quiverkey(q0, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[0].quiverkey(p0, 0.1, 1.03, n, r'{:2.1f} C/m$^2$'.format(n), labelpos="E")
                            axs[0].set_title("$z={:d}$".format(layers[0]), fontsize = self.label_size)
                            axs[0].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[0].set_ylabel("$y$ (unit cells)", fontsize = self.label_size)
                            axs[0].invert_yaxis()

                            m = np.mean(np.hypot(u2, v2))
                            n = np.mean(np.hypot(i2, j2))
                            q1 = axs[1].quiver(X1, Y1, u2, v2, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            p1 = axs[1].quiver(X2, Y2, i2, j2, pivot="mid", width=0.008, headwidth=5, minlength=4, minshaft=3, color="r")
                            axs[1].quiverkey(q1, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[1].quiverkey(p1, 0.1, 1.03, n, r'{:2.1f} C/m$^22$'.format(n), labelpos="E")
                            axs[1].set_title("$z={:d}$".format(layers[1]), fontsize = self.label_size)
                            axs[1].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[1].invert_yaxis()

                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        else:
                            print("\nPLOTTING ERROR: Unsupported number of layers!")
                            pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def AFD_horizontal_domain_vectors(self, layers, mode="a", symmetry=True):

        for i,s in enumerate(self.strain):
            for j,p in enumerate(self.stress):
                for k,f in enumerate(self.field):
                    
                    label = "s{:d}p{:d}f{:d}".format(i,j,k)
                    fplot = os.path.join(self.fplots, label, "T-dep", "AFD"+mode)
                    
                    for _, t in enumerate(self.temp):

                        if symmetry:
                            pname = "AFD" + mode + "_sym_dom_T" + str(int(t)) + ".png"
                        else:
                            pname = "AFD" + mode + "_nosym_dom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)

                        if symmetry:
                            angles = STO_AFD(geom, self.model, mode=mode, angles=True, symmetry=True)
                        else:
                            angles = STO_AFD(geom, self.model, mode=mode, angles=True, symmetry=False)

                        X, Y = np.meshgrid(range(geom.supercell[0]), range(geom.supercell[1]))

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize= self.label_size)
                            plt.ylabel("y (unit cells)", fontsize= self.label_size)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[13,7], sharey=True)
                            if symmetry:
                                fig.suptitle("AFD{:s} Mode Rotation @ $T={:d}$ K (with symmetry corrections)".format(mode, int(t)), fontsize = self.label_size)
                            else:
                                fig.suptitle("AFD{:s} Mode Rotation @ $T={:d}$ K (without symmetry corrections)".format(mode, int(t)), fontsize = self.label_size)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
              
                            m = np.mean(np.hypot(u1, v1))
                            q0 = axs[0].quiver(X, Y, u1, v1, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[0].quiverkey(q0, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[0].set_title("$z={:d}$".format(layers[0]), fontsize = self.label_size)
                            axs[0].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[0].set_ylabel("$y$ (unit cells)", fontsize = self.label_size)
                            axs[0].invert_yaxis()

                            m = np.mean(np.hypot(u2, v2))
                            q1 = axs[1].quiver(X, Y, u2, v2, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[1].quiverkey(q1, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
                            axs[1].set_title("$z={:d}$".format(layers[1]), fontsize = self.label_size)
                            axs[1].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
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
                    fplot = os.path.join(self.fplots, label, "T-dep", "FE")
                    
                    for _, t in enumerate(self.temp):

                        pname = "FEdom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        angles = STO_FE(geom, self.model)

                        X, Y = np.meshgrid(range(geom.supercell[0]), range(geom.supercell[1]))

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            plt.ylabel("$y$ (unit cells)", fontsize = self.label_size)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = angles[:,:,layers[0],0], angles[:,:,layers[0],1]
                            u2, v2 = angles[:,:,layers[1],0], angles[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[13,7], sharey=True)
                            fig.suptitle("FE Mode Amplitude @ $T={:d}$ K".format(int(t)), fontsize = self.label_size)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
              
                            m = np.mean(np.hypot(u1, v1))
                            q0 = axs[0].quiver(X, Y, u1, v1, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[0].quiverkey(q0, 0.9, 1.03, m, '{:3.2f} bohr'.format(m), labelpos='E')
                            axs[0].set_title("$z={:d}$".format(layers[0]), fontsize = self.label_size)
                            axs[0].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
                            axs[0].set_ylabel("$y$ (unit cells)", fontsize = self.label_size)
                            axs[0].invert_yaxis()

                            m = np.mean(np.hypot(u2, v2))
                            q1 = axs[1].quiver(X, Y, u2, v2, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[1].quiverkey(q1, 0.9, 1.03, m, '{:3.2f} bohr'.format(m), labelpos='E')
                            axs[1].set_title("$z={:d}$".format(layers[1]), fontsize = self.label_size)
                            axs[1].set_xlabel("$x$ (unit cells)", fontsize = self.label_size)
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
                    fplot = os.path.join(self.fplots, label, "T-dep", "POL")
                    
                    for _, t in enumerate(self.temp):

                        pname = "POLdom_T" + str(int(t)) + ".png"

                        geom = self.access_geometry(t, p=p, s=s, f=f)
                        pols = STO_POL(geom, self.model)

                        X, Y = np.meshgrid(range(geom.supercell[0]), range(geom.supercell[1]))

                        if len(layers) == 1:

                            plt.figure(pname)
                            u, v = pols[:,:,layers[0],0], pols[:,:,layers[0],1]
                            plt.quiver(u,v, pivot="mid")
                            plt.xlabel("x (unit cells)", fontsize=12)
                            plt.ylabel("y (unit cells)", fontsize=12)
                            plt.tight_layout(pad = self.figure_pad)
                            plt.savefig(os.path.join(fplot, pname))
                            plt.close()

                        if len(layers) == 2:

                            u1, v1 = pols[:,:,layers[0],0], pols[:,:,layers[0],1]
                            u2, v2 = pols[:,:,layers[1],0], pols[:,:,layers[1],1]
                            
                            fig, axs = plt.subplots(1,2, figsize=[13,7], sharey=True)
                            fig.suptitle("Polarization per Unit Cell @ $T={:d}$ K".format(int(t)), fontsize = self.label_size)
                            fig.canvas.set_window_title(pname) 
                            plt.tight_layout(pad = self.figure_pad)
              
                            m = np.mean(np.hypot(u1, v1))
                            q0 = axs[0].quiver(X, Y, u1, v1, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[0].quiverkey(q0, 0.9, 1.03, m, '{:3.2f} C/m^2'.format(m), labelpos='E')
                            axs[0].set_title("$z={:d}$".format(layers[0]), fontsize = self.label_size)
                            axs[0].set_xlabel("x (unit cells)", fontsize=12)
                            axs[0].set_ylabel("y (unit cells)", fontsize=12)
                            axs[0].invert_yaxis()

                            m = np.mean(np.hypot(u2, v2))
                            q1 = axs[1].quiver(X, Y, u2, v2, pivot="mid", width=0.008, headwidth=5, minlength=3, minshaft=3)
                            axs[1].quiverkey(q1, 0.9, 1.03, m, '{:3.2f} C/m^2'.format(m), labelpos='E')
                            axs[1].set_title("$z={:d}$".format(layers[1]), fontsize = self.label_size)
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
                    fplot = os.path.join(self.fplots, label, "T-dep", "POL")
                    
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












