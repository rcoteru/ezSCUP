"""
Several functions regarding the obtention of normal vibrational modes.
"""

# third party imports
import numpy as np

# package imports
from ezSCUP.singlepoint import SPRun
from ezSCUP.geometry import Geometry

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + func finite_hessian()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def finite_hessian(geo, parameter_file, disp=0.001):
    
    pf = parameter_file
    geo.supercell = np.array([1,1,1])
    hessian = np.zeros([(3*geo.nats), (3*geo.nats)])

    for a1 in range(geo.nats):
        for c1 in range(3):
            p1 = 3*a1 + c1
            for a2 in range(geo.nats):
                for c2 in range(3):
                    p2 = 3*a2 + c2

                    if p1 == p2:
                        d0 = geo.displacements[0,0,0,a1,c1] 
                        geo.displacements[0,0,0,a1,c1] = d0 + disp
                        energy_f = SPRun(pf,geo)["total_delta"]
                        geo.displacements[0,0,0,a1,c1] = d0 - disp
                        energy_b = SPRun(pf,geo)["total_delta"]
                        hessian[p1,p2] = (energy_f+energy_b)/(disp**2)
                        geo.displacements[0,0,0,a1,c1] = d0
                    else:
                        d01 = geo.displacements[0,0,0,a1,c1] 
                        d02 = geo.displacements[0,0,0,a2,c2]

                        geo.displacements[0,0,0,a1,c1] = d01 + disp
                        geo.displacements[0,0,0,a2,c2] = d02 + disp
                        energy_pp = SPRun(pf,geo)["total_delta"]

                        geo.displacements[0,0,0,a1,c1] = d01 + disp
                        geo.displacements[0,0,0,a2,c2] = d02 - disp
                        energy_pm = SPRun(pf,geo)["total_delta"]

                        geo.displacements[0,0,0,a1,c1] = d01 - disp
                        geo.displacements[0,0,0,a2,c2] = d02 + disp
                        energy_mp = SPRun(pf,geo)["total_delta"]

                        geo.displacements[0,0,0,a1,c1] = d01 - disp
                        geo.displacements[0,0,0,a2,c2] = d02 - disp
                        energy_mm = SPRun(pf,geo)["total_delta"]

                        hessian[p1,p2]=(energy_pp-energy_pm-energy_mp+energy_mm)/(4*disp**2)

                        geo.displacements[0,0,0,a1,c1] = d01
                        geo.displacements[0,0,0,a2,c2] = d02

    return hessian


def get_normal_modes(masses, hessian):

    M = []
    for m in masses:
        for _ in range(3):    
            M.append(m)
    M = np.diagflat(M)
    Mi = np.linalg.inv(M)

    normHessian = np.matmul(Mi, hessian)
    w, v = np.linalg.eigh(normHessian)

    return w, v


