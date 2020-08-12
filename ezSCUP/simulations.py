"""
Collection of classes to mass-execute ScaleUP simulations.

Classes to execute ScaleUp simulations in a range of 
temperature, strain, stress and electric field settings 
with ease. Each simulation class comes with a parser that 
automates the process of dealing with the output data.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

# third party imports
import numpy as np          # matrix support
import pandas as pd         # .out file loading

# standard library imports
from shutil import move,rmtree      # remove output folder
from pathlib import Path            # general folder management
import os, sys, csv                 # remove files, get pwd
import pickle                       # store parameter vectors
import time                         # check simulation run time

# package imports
from ezSCUP.parsers import REFParser, RestartParser, OutParser
from ezSCUP.geometry import Geometry
from ezSCUP.generators import RestartGenerator
from ezSCUP.structures import FDFSetting
from ezSCUP.handlers import SCUPHandler
import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class MCSimulation():
#   - __init__()
#   - setup()
#   - change_output_folder()
#   - independent_launch()
#
# + class MCConfiguration():
#   - auto_load()
#   - load_unique()
#   - lattice_output()
#
# + class MCSimulationParser() 
#   - __init__()
#   - access()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#####################################################################
## SIMULATIONS (COMPLETE FOLDER)
#####################################################################


class MCSimulation:

    """

    Executes Monte Carlo simulations with the given parameters.
    
    # BASIC USAGE # 
    
    Creates an output folder in the current directory.
    All the simulation data for each configuration is 
    conveniently stored in subfolders within it:
        
        output / [scale-up system_name].c[n]

    where n is the configuration number. That is, an 8 digit
    integer that specifies the index for each parameter in the
    given simulation, with two digits each in the following order:

        n = [temp][stress][strain][field]
    
    for example, in a simulation with temp=[20,40,60,80] all confs
    simulated at 40K will be stored in folders named

        output / [scale-up system_name].c01******

    and so on. The information about the parameters for the simulation
    run is stored in the file:

        output / simulation.info # formated as a pickle file

    In order to access all the simulation output for any given 
    configuration, refer to the class MCSimulationParser in this
    same module.

    # SIMULATION PARAMETER SPECIFICATION #

    - Temperature (temp):
        List of temperatures, in K
            i.e:temp = np.linspace(20,100,5)
                temp = [27, 45, 36]

    - External Stress (stress):
        List of stress vectors in Voigt notation, in GPa
            i.e:stress = [
                [10., 0., 0., 0., 0., 0.],
                [0., 10., 0., 0., 0., 0.],
                [0., 0., 10., 0., 0., 0.]
            ]   # 10 GPa strains in each direction

    - Strains (strain):
        List of strain vectors in Voigt notation, in percentual change
            i.e:strain = [
                [+0.02, +0.02, 0., 0., 0., 0.],
                [+0.00, +0.00, 0., 0., 0., 0.],
                [-0.02, -0.02, 0., 0., 0., 0.]
            ]   # +-2% and 0% cell strain in the x and y direction

    - Static Electric Field (field):
        List of electric field vectors, in V/m
            i.e:field = [
                [1e9, 0., 0.]
            ]   # 1e9 V/m = 1V/nm in the x direction  
    
    Attributes:
    ----------

     - fdf (string): base fdf file name
     - name (string): default fdf's system_name

     - temp (array): temperature vectors (K) 
     - stress (array): stress vectors (Gpa)
     - strain (array): strain vectors (% change) 
     - field (array): electric field vectors (V/m)


    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def setup(
        self, fdf, temp, stress=None, strain=None, field=None, 
        supercell = None, species=None, nats=None,
        output_folder="output" 
        ):

        """
        
        Sets up everything

        Parameters:
        ----------

        - fdf  (string): common fdf base file for all simulations.

        - temp (list): list of temperatures, required.
        - stress (list): list of stress vectors, if needed.
        - strain (list): list of strain vectors, if needed.
        - field (list): list of field vectors, if needed.

        """

        # first and foremost, check if a valid 
        # ScaleUP executable has been configured.
        if cfg.SCUP_EXEC == None or not os.path.exists(cfg.SCUP_EXEC):

            print("""
            WARNING: No valid executable detected

            A path for a valid ScaleUP executable
            must be provided before any simulations
            are carried out.

            In order to do this, include the lines

            import ezSCUP.settings as cfg
            cfg.SCUP_EXEC=[path_to_exec]

            at the beginning of your script.
            """)

            raise ezSCUP.exceptions.NoSCUPExecutableDetected

        self.fdf = fdf # load the main fdf file
        self.sim = SCUPHandler(scup_exec=cfg.SCUP_EXEC)
        self.sim.load(self.fdf)

        self.supercell = supercell
        self.species = species
        self.nats = nats

        self.output_folder = output_folder # current output folder

        self.temp = np.array(temp) # temperature vector, required
        
        if stress == None: # stress vector, optional
            self.stress = [np.zeros(6)]
        else:
            self.stress = [np.array(p, dtype=np.float64) for p in stress]

        if strain == None: # strain vector list, optional
            self.strain = [np.zeros(6)]
        else:
            self.strain = [np.array(s, dtype=np.float64) for s in strain]
        
        if field == None: # electric field vector list, optional
            self.field = [np.zeros(3)]
        else:
            self.field = [np.array(f, dtype=np.float64) for f in field]

        # get the current path
        self.current_path = os.getcwd()

        # create output directory
        try:
            self.main_output_folder = os.path.join(self.current_path, self.output_folder)
            os.makedirs(self.main_output_folder)
        except FileExistsError: # check whether directory already exists
            if cfg.OVERWRITE:
                print("""
                Found already existing output 
                folder named "{}",
                all its contents are now lost.
                Reason: OVERWRITE set to True.
                """.format(self.output_folder))
                rmtree(self.main_output_folder)
                print("")
                pass
            else:
                print("""
                Found already existing output 
                folder named "{}",
                skipping simulation process.
                Reason: OVERWRITE set to False.
                """.format(self.output_folder))
                raise ezSCUP.exceptions.PreviouslyUsedOutputFolder()

        # adjust the supercell as needed
        if self.supercell != None:
            self.sim.settings["supercell"] = [list(self.supercell)]
        else:
            self.supercell = self.sim.settings["supercell"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # modify FDF settings according to ezSCUP.settings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        if cfg.MC_STEPS != None: # number of total MC steps
            self.sim.settings["mc_nsweeps"].value = int(cfg.MC_STEPS)
            self.mc_steps = int(cfg.MC_STEPS)
        else:
            self.mc_steps = self.sim.settings["mc_nsweeps"].value

        if cfg.MC_STEP_INTERVAL != None: # interval between partials
            self.sim.settings["n_write_mc"].value = int(cfg.MC_STEP_INTERVAL)
            self.mc_step_interval = int(cfg.MC_STEP_INTERVAL)
        else:
            self.mc_step_interval = self.sim.settings["n_write_mc"].value

        if cfg.MC_MAX_JUMP != None: # max jump distance 
            self.sim.settings["mc_max_step_d"].value = float(cfg.MC_MAX_JUMP)
            self.mc_max_jump = float(cfg.MC_MAX_JUMP)
        else:
            self.mc_max_jump = self.sim.settings["mc_max_step_d"].value

        if cfg.LATTICE_OUTPUT_INTERVAL != None:
            self.sim.settings["print_std_lattice_nsteps"].value = int(cfg.LATTICE_OUTPUT_INTERVAL)
            self.lat_output_interval = int(cfg.LATTICE_OUTPUT_INTERVAL)
        else:
            self.lat_output_interval = self.sim.settings["print_std_lattice_nsteps"].value
        
        if cfg.FIXED_STRAIN_COMPONENTS != None:
            if len(cfg.FIXED_STRAIN_COMPONENTS) != 6:
                raise ezSCUP.exceptions.InvalidFDFSetting
            setting = []
            for s in list(cfg.FIXED_STRAIN_COMPONENTS):
                if not isinstance(s, bool):
                    raise ezSCUP.exceptions.InvalidFDFSetting
                if s:
                    setting.append("T")
                else:
                    setting.append("F")
            print(setting)
            self.sim.settings["fix_strain_component"] = [setting]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # strains setup (require a restart file generator)
        if strain != None:

            if supercell == None or nats == None or species == None:
                raise ezSCUP.exceptions.MissingRequiredArguments(
                """
                In order to simulate strains you must also provide
                the supercell, species and number of atoms, as the
                creation of .restart files is needed.
                """
                )

            self.sim.settings["geometry_restart"] = FDFSetting("temporary.restart")
            self.generator = RestartGenerator(self.supercell, self.species, self.nats)

        # print common simulation settings
        print("\nCurrent FDF settings:")
        self.sim.print_all()
        print("")

        # load simulation name
        self.name = self.sim.settings["system_name"].value
        
        # save simulation setup file 
        print("\nSaving simulation setup file... ")

        setup = {
            "name": self.name,
            "supercell": self.supercell,
            "species": self.species,
            "nats": self.nats,
            "temp": self.temp,
            "strain": self.strain,
            "stress": self.stress,
            "field": self.field,

            "mc_steps": self.mc_steps,
            "mc_step_interval": self.mc_step_interval,
            "mc_max_jump": self.mc_max_jump,
            "lat_output_interval": self.lat_output_interval,

            "fixed_strain_components": cfg.FIXED_STRAIN_COMPONENTS,
            }

        simulation_setup_file = os.path.join(self.main_output_folder, cfg.SIMULATION_SETUP_FILE)

        with open(simulation_setup_file, "wb") as f:
            pickle.dump(setup, f)

        print("\n Simulation run has been properly configured.")
        print("You may now proceed to launch it.\n")

        self.SETUP = True # simulation run properly setup

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def change_output_folder(self, output_folder):

        # get the current path
        self.current_path = os.getcwd()

        previous = self.output_folder

        # create output directory
        print("Creating output folder...")
        try:
            main_output_folder = os.path.join(self.current_path, output_folder)
            os.makedirs(main_output_folder)
            self.output_folder = output_folder
            self.main_output_folder = main_output_folder

        except FileExistsError: # check whether directory already exists
            if cfg.OVERWRITE:
                print("""
                Found already existing output 
                folder named "{}",
                all its contents are now lost.
                Reason: OVERWRITE set to True.
                """.format(self.output_folder))
                rmtree(self.main_output_folder)
                print("")
                pass
            else:
                print("""
                Found already existing output 
                folder named "{}", aborting folder swap.
                Reason: OVERWRITE set to False.
                """.format(self.output_folder))
                raise ezSCUP.exceptions.PreviouslyUsedOutputFolder()

        # save simulation setup file 
        print("\nSaving simulation setup file... ")

        setup = {
            "name": self.name,
            "supercell": self.supercell,
            "species": self.species,
            "nats": self.nats,
            "temp": self.temp,
            "strain": self.strain,
            "stress": self.stress,
            "field": self.field,

            "mc_steps": self.mc_steps,
            "mc_step_interval": self.mc_step_interval,
            "mc_max_jump": self.mc_max_jump,
            "lat_output_interval": self.lat_output_interval,

            "fixed_strain_components": cfg.FIXED_STRAIN_COMPONENTS,
            }

        simulation_setup_file = os.path.join(self.main_output_folder, cfg.SIMULATION_SETUP_FILE)

        with open(simulation_setup_file, "wb") as f:
            pickle.dump(setup, f)

        print('\n Output folder swap from "{}" to "{}" complete!\n'.format(previous, self.output_folder))

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def independent_launch(self, start_geo = None):

        """
        
        Start a simulation run with all the possible combinations 
        of the given parameters. 

        Parameters:
        ----------

        - fdf  (string): common fdf base file for all simulations.

        - temp (list): list of temperatures, required.
        - stress (list): list of stress vectors, if needed.
        - strain (list): list of strain vectors, if needed.
        - field (list): list of field vectors, if needed.

        """
        
        # simulation counters
        total_counter   =  0

        # total number of simulations 
        nsims = self.temp.size*len(self.strain)*len(self.field)*len(self.stress)
        
        # starting time of the simulation process
        main_start_time = time.time()

        print("\nStarting calculations...\n")
        for t in self.temp:
            temp_counter = np.where(self.temp == t)[0][0]
            for p in self.stress:
                stress_counter = [np.array_equal(p,x) for x in self.stress].index(True)
                for s in self.strain:
                    strain_counter = [np.array_equal(s,x) for x in self.strain].index(True)
                    for f in self.field:
                        field_counter = [np.array_equal(f,x) for x in self.field].index(True)

                        # update simulation counter
                        total_counter += 1

                        # starting time of the current configuration
                        conf_start_time = time.time()

                        print("##############################")
                        print("Configuration " + str(total_counter) + " out of " + str(nsims))
                        print("Temperature:",   str(t),"K")
                        print("Stress:",        str(p),"GPa")
                        print("Strain:",        str(s), r"% change")
                        print("Electric Field:",str(f), "V/m")
                        print("##############################")

                        # file base name
                        sim_name = self.name + "T{:d}".format(int(t))
                        self.sim.settings["system_name"].value = sim_name

                        # configuration name
                        conf_name = "c{:02d}{:02d}{:02d}{:02d}".format(temp_counter,
                            stress_counter, strain_counter, field_counter)

                        # subfolder name
                        subfolder_name = self.name + "." + conf_name

                        # modify target temperature
                        self.sim.settings["mc_temperature"].value = t

                        # modify target stress, if needed
                        if self.stress != None: 
                            self.sim.settings["external_stress"] = [p]

                        # set target strain, if needed
                        if self.strain != None:
                            self.generator.strains = s
                            self.generator.write("temporary.restart")

                        # modify target field, if needed
                        if self.field != None: 
                            self.sim.settings["static_electric_field"] = [f]

                        # define human output filename
                        output = sim_name + ".out"

                        # simulate the current configuration
                        self.sim.launch(output_file=output)

                        # move all the output to its corresponding folder
                        configuration_folder = os.path.join(self.main_output_folder, subfolder_name)
                        os.makedirs(configuration_folder)

                        files = os.listdir(self.current_path)
                        for f in files:
                            if sim_name in f:
                                move(f, configuration_folder)
                        
                        # finish time of the current conf
                        conf_finish_time = time.time()

                        conf_time = conf_finish_time-conf_start_time

                        print("\n Configuration finished! (time elapsed: {:.3f}s)".format(conf_time))
                        print("All files stored in output/" + subfolder_name + " succesfully.\n")


        # cleanup
        if self.strain != None:
            os.remove("temporary.restart")

        main_finished_time = time.time()
        main_time = main_finished_time - main_start_time

        print("Simulation process complete!")
        print("Total simulation time: {:.3f}s".format(main_time))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class MCConfiguration:

    """

    WARNING: This class is NOT meant to be manually 
    created, its only purpose is to be returned by 
    the MCConfigurationParser class.

    Parses all the data from a previously simulated 
    configuration, providing easy access to all its output. 
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration (subfolder) in the
    output folder. By default this class starts empty, until a folder
    is loaded with the load() method.

    Basic information about the simulation such as supercell shape,
    original lattice constants, elements information and number of
    atoms in the unit cell are accessible via attributes.

    The MC .partial file data is averaged out and stored in self.strains
    and self.cells. More on cell information storage in ezSCUP.structures.

    All the output file information is accessed through the output_file()
    method, which returns a pandas Dataframe with the lattice ("LT:") data.

    # ACCESSING CELL DATA #

    In order to access cell data after loading the cell data in the class,
    for example after acessing via the MCConfigurationParser class, just do:

        sim = MCConfigurationParser()           # instantiate the parser class
        config = sim.access(t, s=s, [...])      # access the configuration (this class)
        cell = config.geo.cells[x,y,z]          # access the desired cell
        cell.positions["element_label"]         # position data by label
        cell.displacements["element_label"]     # displacement data by label

    where x, y and z is the position of the desired cell in the supercell.
    This will return a "Cell" class with an attribute "displacements", 
    a dictionary with the displacement vector (average of partial .restarts)
    for each element label, and another "positions" dictionary, with the 
    position vectors (from REF file) for each label. (more within the next 
    section)

    # ELEMENT LABELING #

    By default, SCALE-UP does not label elements in the output beside a 
    non-descript number. This programs assigns a label to every atom in
    order to easily access their data from dictionaries.

    Suppose you have an SrTiO3 cell, only three elements but five atoms.
    Then the corresponding labels would be ["Sr","Ti","O1","O2","O3"].

    Attributes:
    ----------

     - name (string): file base name
     - folder_name (string): configuration folder (name)
     - folder_path (string): configuration folder (full path)
     - current_directory (sring): main script directory
     - partials (list): partial files in the folder (full path)

     - nmeas (int): number of measurements above the threshold
     - total_steps (int): number of MC steps taken
     - step_threshold: only pick partials with higher step 
        than this. DEFAULT: cfg.MC_EQUILIBRATION_STEPS
     
     - geo (Geometry): class containing the system's geometry (ezSCUP.geometry)
    """ 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def auto_load(self, folder_name, base_sim_name, config_data):

        """
        
        Load the given configuration folder's data.

        Parameters:
        ----------

        - folder_name  (string): configuration folder name,
            usually [ScaleUP system_name].c[configuration number]

        - base_sim_name (list): basename of the ScaleUP files,
            usually [ScaleUP system_name].T[temperature in integer format]

        """

        # load configuration parameters
        self.temp =     config_data[0]
        self.pressure = config_data[1]
        self.strain =   config_data[2]
        self.field =    config_data[3]

        # load file and folder names
        self.name = base_sim_name
        self.folder_name = folder_name
        self.current_directory = os.getcwd()

        folder = os.path.join(self.current_directory, "output", folder_name)
        self.folder_path = folder

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # get partials, number of MC steps, etc
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # partial .restart files
        partials = [k for k in os.listdir(folder) if 'partial' in k]
        partials = sorted([k for k in partials if '.restart' in k])
        self.partials = [os.path.join(self.folder_path, p) for p in partials]
        # get total number of MC steps
        self.total_steps = max([int(p[len(base_sim_name)+10:-8]) for p in partials])
        self.step_threshold = cfg.MC_EQUILIBRATION_STEPS 
        # check if any measurements are taken into consideration
        if (self.total_steps <= self.step_threshold):
            print("Step threshold {:d} greater than total steps ({:d}):".format(self.step_threshold, self.total_steps))
            print(r"Reducing to 20% of total steps.")
            self.step_threshold = int(0.2*self.total_steps)
        # selects only partials above self.step_threshold steps
        step_filter = [int(p[len(base_sim_name)+10:-8]) > self.step_threshold for p in partials]
        partials = [p for i, p in enumerate(partials) if step_filter[i]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        reference_file = os.path.join(folder, base_sim_name + "_FINAL.REF")

        self.geo = Geometry(reference_file)
        self.geo.load_equilibrium_displacements(partials)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def load_unique(self, base_sim_name):

        """
        
        Loads a single point

        """

        self.current_directory = os.getcwd()
        self.name = base_sim_name

        self.outp = OutParser()

        # loads basic filename information
        self.name = base_sim_name
        self.folder_name = None
        self.current_directory = os.getcwd()
        self.folder_path = self.current_directory

        # no partials loaded
        self.partials = []

        reference_file = os.path.join(self.name + "_FINAL.REF")
        restart_file = os.path.join(self.name + "_FINAL.restart")

        self.geo = Geometry(reference_file)
        self.geo.load_restart(restart_file)
        

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def lattice_output(self):

        #TODO REMAKE THIS
        
        """

        Load the given configuration lattice data from its output file.

        Return:
        ----------
            - a pandas Dataframe with the lattice output data.
        """

        self.step_threshold = cfg.MC_EQUILIBRATION_STEPS

        out_file = os.path.join(self.folder_path, self.name + ".out")
        self.outp.load(out_file)

        ldata = self.outp.lattice_data.copy(deep=True)
        ldata = ldata[ldata.index >= self.step_threshold]

        return ldata

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
   

class MCSimulationParser:

    """

    Parses all the data from a previously simulated configuration, 
    providing easy access to all its output. 
    
    # BASIC USAGE # 
    
    Reads simulation data from a given configuration (subfolder) in the
    output folder. By default this class starts empty, until a folder
    is loaded with the load() method.

    Basic information about the simulation such as supercell shape,
    original lattice constants, elements information and number of
    atoms in the unit cell are accessible via attributes.

    The MC .partial file data is averaged out and stored in self.strains
    and self.cells. More on cell information storage in ezSCUP.structures.

    All the output file information is accessed through the output_file()
    method, which returns a pandas Dataframe with the lattice ("LT:") data.

    Attributes:
    ----------

     - name (string): simualtion file base name

     - temp (array): temperature vectors (K) 
     - stress (array): stress vectors (Gpa)
     - strain (array): strain vectors (% change) 
     - field (array): electric field vectors (V/m)

     - parser (MCConfiguration): last accessed configuration

    """

    ########################
    #      ATTRIBUTES      #
    ########################

    parser = MCConfiguration()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, output_folder="output"):

        # check if there's an output folder
        if os.path.exists(os.path.join(os.getcwd(), "output")):
            
            # first, load the simulation.info file from the output folder
            
            # get the current path
            self.current_path = os.getcwd()

            # main output folder
            self.main_output_folder = os.path.join(self.current_path, output_folder)

            # simulation setup file 
            self.simulation_setup_file = os.path.join(self.main_output_folder, cfg.SIMULATION_SETUP_FILE)

            # load simulation setup file 
            with open(self.simulation_setup_file, "rb") as f:
                setup = pickle.load(f) 

            # load run information
            self.name      = setup["name"] 
            self.supercell = setup["supercell"]
            self.elements  = setup["elements"]
            self.nats      = setup["nats"]

            # load simulation parameters
            self.temp   = setup["temp"] 
            self.stress = setup["stress"] 
            self.strain = setup["strain"] 
            self.field  = setup["field"]   

        else:
            print('Cannot find output folder "{}", exiting.'.format(output_folder))
            raise ezSCUP.exceptions.OutputFolderDoesNotExist   

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def print_simulation_setup(self):
        
        # TODO
        
        pass

    def access(self, t, p=None, s=None, f=None):

        """

        Accesses configuration data for the specified parameters.

        Parameters:
        ----------

        - t (float): Temperature (compulsory)
        - p (array): Pressure (optional)
        - s (array): Strain (optional)
        - f (array): Electric Field (optional)

        Return:
        ----------
            - The MCConfiguration object corresponding to the desired configuration.

        """

        # stress vector, optional
        if p == None:
            p = np.zeros(6)
        else:
            p = np.array(p)

        # strain vector list, optional
        if s == None:
            s = np.zeros(6)
        else:
            s = np.array(s)
        
        # electric field vector list, optional
        if f == None:
            f = np.zeros(3)
        else:
            f = np.array(f)

        # obtain index of desired parameters
        try: 
            t_index = np.where(self.temp == t)[0][0]
            p_index = [np.array_equal(p,x) for x in self.stress].index(True)
            s_index = [np.array_equal(s,x) for x in self.strain].index(True)
            f_index = [np.array_equal(f,x) for x in self.field].index(True)
        except:
            raise ezSCUP.exceptions.InvalidMCConfiguration(
                "The configuration has not been simulated."
            )

        # get configuration name
        conf_name = "c{:02d}{:02d}{:02d}{:02d}".format(t_index, 
            p_index, s_index, f_index)

        # subfolder name
        subfolder_name = self.name + "." + conf_name
        sim_name = self.name + "T{:d}".format(int(t))

        self.parser.auto_load(subfolder_name, sim_name, config_data=[t, p, s, f])
        
        return self.parser

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        