"""
Classes to mass-execute SIESTA geometry optimizations.
"""

# third party imports
from ezSCUP.structures import Geometry

# standard library imports
from shutil import move,rmtree,copy # remove output folder
from pathlib import Path            # general folder management
import os, sys, csv                 # remove files, get pwd


SCUP_EXEC = os.getenv("SCUP_EXEC", default = None) 

if not os.path.exists(SCUP_EXEC):
    print("WARNING: SIESTA executable provided does not exist.")
    #exit

def log(log_file, message):

    with open(log_file, "a+") as f:
        # Move read cursor to the start of file.
        f.seek(0)
        # If file is not empty then append '\n'
        data = f.read(100)
        if len(data) > 0 :
            f.write("\n")
        # Append text at the end of file
        f.write(message)


class FDFManager():

    common_params = {
        "system_name":          ["SCUP_run"],
        "parameter_file":       [None],
        "run_mode":             [None],
        "supercell":            [[1, 1, 1]],
        "no_electron":          [".true."],
        "fix_strain_component": [["F","F","F","F","F","F"]],
        # printouts
        "print_std_lattice_nsteps":     [1],
        "print_std_energy":             [".true."],
        "print_std_av_energy":          [".true."],
        "print_std_delta_energy":       [".true."],
        "print_std_polarization":       [".true."],
        "print_std_av_polarization":    [".true."],
        "print_std_strain":             [".true."],
        "print_std_av_energy":          [".true."],
        "print_std_temperature":        [".true."],
    }

    SP_params = {
        "run_mode":     ["single_point"],
    }

    OPT_params = {
        "maximumgeomiter":      [1000],
        "forcethreshold":       [0.01, "eV/Ang"],
        "opt_forcefactor":      [1.0],
        "opt_stressfactor":     [10.0],
        "opt_maximum_step":     [0.01, "bohr"]
    }

    MC_params = {
        "mc_temperature":    [298.15, "kelvin"],
        "mc_annealing_rate": [1],
        "mc_strains":        [".true."],
        "mc_nsweeps":        [1000],
        "mc_max_step":       [0.5, "ang"],
        "n_write_mc":        [20],
        "print_justgeo":     [".true."]
    }

    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, runtype:str = "SP"):
        self.reset(runtype=runtype)

    def reset(self, runtype="SP"):

        self.settings = self.common_params

        if   runtype == "SP":
            self.settings["run_mode"] = ["single_point"]
        elif runtype == "OPT":
            self.settings["run_mode"] = ["optimization"]
            self.settings.update(self.OPT_params)
        elif runtype == "MC":
            self.settings["run_mode"] = ["monte_carlo"]
            self.settings.update(self.MC_params)
        else:
            pass

    def fix_strain_components(self, str_comps:list):

        if len(str_comps) != 6:
            print("ERROR: Invalid strain restrictions.")
            exit

        for r in str_comps:
            if type(r) is not bool:
                print("ERROR: Invalid strain restrictions.")
                exit

        str_set = []
        for r in str_comps:
            if r:
                str_set.append("T")
            else:
                str_set.append("F")

        self.settings["fix_strain_component"] = [str_set]

    def read_fdf(self, fdf_file:str):

        # empty previous settings
        self.reset()

        with open(fdf_file, "r") as f:

            lines = f.readlines()

            # do a general cleanup
            lines = [l.strip() for l in lines]          # remove trailing spaces
            lines = [l.split("#",1)[0] for l in lines]  # removes comments
            lines = [l for l in lines if l]             # remove empty lines
            lines = [l.lower() for l in lines]          # make everything lowercase
            lines = [l.split() for l in lines]          # split lines into lists

        i = 0
        block = None
        blockname = None

        while i < len(lines): 
            
            line = lines[i]
            
            # check if its a block setting
            if line[0] == r"%block":

                block = []
                blockname = line[1]
                
                i+=1
                line = lines[i]

                while line[0] != r"%endblock":
                    block.append(line)
                    i += 1
                    line = lines[i]

                self.settings[blockname] = block

            # if its not, then just read it
            else:
                if len(line) == 1:
                    # setting without value = true
                    self.settings[line[0]] = [".true."]
                else:
                    self.settings[line[0]] = line[1:]

            i += 1
        
        pass

    def write_fdf(self, fname:str):

        f = open(fname, "w")

        for k in self.settings:

            if isinstance(self.settings[k][0], list):

                f.write(r"%block " + k + "\n")

                # turn array into string
                string = ""
                for row in self.settings[k]:
                    for n in row:
                        string += str(n) + " "
                    string += "\n"

                f.write(string)
                f.write(r"%endblock " + k + "\n")
            
            else:
                
                line = k + " " + " ".join([str(el) for el in self.settings[k]]) + "\n" 
                f.write(line)

            f.write("\n")
                
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                         simple jobs                           #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def single_point_run(folder, geom:Geometry,
    fdf:FDFManager = FDFManager(runtype="SP")):

    # create working directory
    original_dir = os.getcwd()
    working_dir = os.path.abspath(folder)

    try:
        os.makedirs(working_dir)
    except:
        pass

    # move to directory 
    os.chdir(working_dir)

    # copy model to folder
    _, model_fname = os.path.split(geom.model.file)
    copy(geom.model.file, working_dir)

    # create input geometry file
    restart_fname = "ezSCUP_start_geom.restart"
    geom.write_restart(restart_fname)

    # edit required fdf settings
    fdf.settings["supercell"]        = [list(geom.sc)]
    fdf.settings["parameter_file"]   = [model_fname]
    fdf.settings["geometry_restart"] = [restart_fname]

    # create input settings file
    fdf_file  = "ezSCUP_input.fdf"
    fdf.write_fdf(os.path.join(working_dir, fdf_file))

    # run the command
    command = f"{SCUP_EXEC} < {fdf_file} > SCUP_output.txt"
    os.system(command)

    # remove model file
    os.remove(os.path.join(working_dir, model_fname))

    # go back to 505
    os.chdir(original_dir)

    return True

def optimization_run(folder, geom:Geometry, max_its:int=1000, 
    fdf:FDFManager = FDFManager(runtype="OPT")) -> None:

    # create working directory
    original_dir = os.getcwd()
    working_dir = os.path.abspath(folder)

    try:
        os.makedirs(working_dir)
    except:
        pass

    # move to directory 
    os.chdir(working_dir)

    # copy model to folder
    _, model_fname = os.path.split(geom.model.file)
    copy(geom.model.file, working_dir)

    # create input geometry file
    restart_fname = "ezSCUP_start_geom.restart"
    geom.write_restart(restart_fname)

    # edit required fdf settings
    fdf.settings["supercell"]        = [list(geom.sc)]
    fdf.settings["parameter_file"]   = [model_fname]
    fdf.settings["geometry_restart"] = [restart_fname]
    fdf.settings["maximumgeomiter"]  = [max_its]

    # create input settings file
    fdf_file  = "ezSCUP_input.fdf"
    fdf.write_fdf(os.path.join(working_dir, fdf_file))

    # run the command
    command = f"{SCUP_EXEC} < {fdf_file} > SCUP_output.txt"
    os.system(command)

    # remove model file
    os.remove(os.path.join(working_dir, model_fname))

    # go back to 505
    os.chdir(original_dir)

    return True

def montecarlo_run(folder, geom:Geometry, temp:float = 298.15, 
    nsteps:int = 1000, fdf:FDFManager = FDFManager(runtype="OPT")) -> None:

    # create working directory
    original_dir = os.getcwd()
    working_dir = os.path.abspath(folder)

    try:
        os.makedirs(working_dir)
    except:
        pass

    # move to directory 
    os.chdir(working_dir)

    # copy model to folder
    _, model_fname = os.path.split(geom.model.file)
    copy(geom.model.file, working_dir)

    # create input geometry file
    restart_fname = "ezSCUP_start_geom.restart"
    geom.write_restart(restart_fname)

    # edit required fdf settings
    fdf.settings["supercell"]        = [list(geom.sc)]
    fdf.settings["parameter_file"]   = [model_fname]
    fdf.settings["geometry_restart"] = [restart_fname]
    fdf.settings["mc_nsweeps"]       = [nsteps]
    fdf.settings["mc_temperature"]   = [temp, "kelvin"],

    # create input settings file
    fdf_file  = "ezSCUP_input.fdf"
    fdf.write_fdf(os.path.join(working_dir, fdf_file))

    # run the command
    command = f"{SCUP_EXEC} < {fdf_file} > SCUP_output.txt"
    os.system(command)

    # read partial files ?
    # create equilibrium geometry?

    # remove model file
    os.remove(os.path.join(working_dir, model_fname))

    # go back to 505
    os.chdir(original_dir)

    return True
