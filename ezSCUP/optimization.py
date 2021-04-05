"""
Classes to mass-execute SCALE-UP Monte Carlo simulations in a range of  
temperatures, strain, stress and electric field settings 
with ease.
"""

# third party imports
import numpy as np          # matrix support
import pandas as pd         # .out file loading

# standard library imports
from shutil import move,rmtree,copy # remove output folder
from pathlib import Path            # general folder management
import os, sys, csv                 # remove files, get pwd
import pickle                       # store parameter vectors
import time                         # check simulation run time
import re                           # regular expressions

# package imports
from ezSCUP.handlers    import CGD_SCUPHandler, FDFSetting
from ezSCUP.singlepoint import SPRun
from ezSCUP.geometry    import Geometry

from ezSCUP.srtio3.models import STO_JPCM2013

import ezSCUP.settings as cfg
import ezSCUP.exceptions

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MODULE STRUCTURE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# + class CGDSimulationParser() 
#   - __init__()
#   - access()
#   - print_simulation_setup()
#
# + class CGDSimulation():
#   - __init__()
#   - setup()
#   - change_output_folder()
#   - independent_launch()
#   - sequential_launch_by_temperature()
#   - sequential_launch_by_strain()
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class CGDSimulationParser:

    """

    Provides easy access to the data from a previous simulation.
    
    Attributes:
    ----------

     - name (string): simulation file base name
     - supercell (array): supercell shape
     - species (list): atomic species within the supercell
     - nats (int): number of atoms per unit cell

     . main_output_folder (string):
     - simulation_setup_file(string): 
     - mc_steps (int): MC steps per simulation.
     - mc_step_interval (int): MC steps between partial .restarts
     - mc_equilibration_steps (int): equilibration steps for the
    calculated equilibirum geometry.
     - mc_max_jump (float): MC max jump distance, in Angstrom.
     - lat_output_interval (int): MC step interval between lattice
     data entries.
     - fixed_strain_components (list): fixed strain components.

     - temp (array): temperature vectors (K) 
     - stress (array): stress vectors (Gpa)
     - strain (array): strain vectors (% change) 
     - field (array): electric field vectors (V/m)

    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, output_folder="output"):

        # check if there's an output folder
        if os.path.exists(os.path.join(os.getcwd(), output_folder)):
            
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

            # check simulation type
            if setup["run_type"] != "cgd":
                print("ERROR: Wrong Simulation type, quitting.") 

            # load simulation run info
            self.name      = setup["name"] 
            self.supercell = setup["supercell"]
            try:
                self.model = setup["model"]
            except:
                self.model = STO_JPCM2013

            self.cgd_max_iter           = setup["cgd_max_iter"]
            self.cgd_force_threshold    = setup["cgd_force_threshold"]
            self.cgd_force_factor       = setup["cgd_force_factor"]
            self.cgd_stress_factor      = setup["cgd_stress_factor"]
            self.cgd_max_step           = setup["cgd_max_step"]

            self.fixed_strain_components = setup["fixed_strain_components"]

            self.stress = setup["stress"] 
            self.strain = setup["strain"] 
            self.field  = setup["field"]   

        else:
            print('Cannot find output folder "{}", exiting.'.format(output_folder))
            raise ezSCUP.exceptions.OutputFolderDoesNotExist()   

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def get_location(self, p=None, s=None, f=None):

        """

        Return the location (folder) and base filename of a given configuration.

        Parameters:
        ----------

        - p (array): Pressure (optional)
        - s (array): Strain (optional)
        - f (array): Electric Field (optional)

        Return:
        ----------
            - Folder and base filename of the requested configuration.

        """

        # stress vector, optional
        if p is None:
            p = np.zeros(6)
        else:
            p = np.array(p)

        # strain vector list, optional
        if s is None:
            s = np.zeros(6)
        else:
            s = np.array(s)
        
        # electric field vector list, optional
        if f is None:
            f = np.zeros(3)
        else:
            f = np.array(f)

        # obtain index of desired parameters
        try: 
            p_index = [np.array_equal(p,x) for x in self.stress].index(True)
            s_index = [np.array_equal(s,x) for x in self.strain].index(True)
            f_index = [np.array_equal(f,x) for x in self.field].index(True)
        except:
            raise ezSCUP.exceptions.InvalidMCConfiguration(
                "The requested configuration has not been simulated."
            )

        # obtain configuration name
        conf_name = "c{:02d}{:02d}{:02d}".format(p_index, s_index, f_index)

        # folder and file naming
        sim_name = self.name + "_CGD"
        subfolder_name = self.name + "." + conf_name
        folder = os.path.join(self.main_output_folder, subfolder_name)

        return folder, sim_name

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def access_energy(self, p=None, s=None, f=None):

        """

        Access the energy of the corresponding configuration.

        Parameters:
        ----------

        - p (array): Pressure (optional)
        - s (array): Strain (optional)
        - f (array): Electric Field (optional)

        Return:
        ----------
            - A Geometry object for the corresponding configuration.

        """

        # folder and file naming
        folder, _ = self.get_location(p, s, f)

        energy_file = os.path.join(folder, "energy.pickle")

        with open(energy_file, "rb") as f:
            energy = pickle.load(f)
        
        return energy

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def access_geometry(self, p=None, s=None, f=None):

        """

        Access the equilibrium geometry of the corresponding configuration.

        Parameters:
        ----------

        - p (array): Pressure (optional)
        - s (array): Strain (optional)
        - f (array): Electric Field (optional)

        Return:
        ----------
            - A Geometry object for the corresponding configuration.

        """

        # folder and file naming
        folder, sim_name = self.get_location(p, s, f)

        reference_file = os.path.join(folder, sim_name + "_FINAL.REF")
        restart_file = os.path.join(folder, sim_name + "_FINAL.restart")

        geo = Geometry(self.supercell, self.model["species"], self.model["nats"])
        geo.load_reference(reference_file)
        geo.load_restart(restart_file)
        
        return geo

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
    def access_lattice_output(self, p=None, s=None, f=None):

        """

        Access the lattice output data of the corresponding configuration.

        Parameters:
        ----------

        - p (array): Pressure (optional)
        - s (array): Strain (optional)
        - f (array): Electric Field (optional)

        Return:
        ----------
            - A pandas Dataframe with the lattice output corresponding to 
            the desired configuration.

        """

        # folder and file naming
        folder, sim_name = self.get_location(p, s, f)
        output_file = os.path.join(folder, sim_name + ".out")

        # create a temporary file that
        # contains only the lattice data
        f = open(output_file, "r")
        temp = open(".temp_re_search.txt", "w")
        for line in f:
            if re.search(cfg.LT_SEARCH_WORD, line):
                temp.write(line)
        temp.close()  
        f.close()

        # read the lattice data
        lattice_data = pd.read_csv(".temp_re_search.txt", delimiter=r'\s+')
        del lattice_data["LT:"]
        lattice_data.set_index('Iter', inplace=True)

        # remove temporary file
        os.remove(".temp_re_search.txt")

        return lattice_data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# ================================================================= #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ================================================================= #

class CGDSimulation:

    """

    Executes Monte Carlo simulations with the given parameters.
    
    # BASIC USAGE # 
    
    Creates an output folder in the current directory.
    All the simulation data for each configuration is 
    conveniently stored in subfolders within it:
        
        output / [scale-up system_name].c[n]

    where n is the configuration number. That is, a 6 digit
    integer that specifies the index for each parameter in the
    given simulation, with two digits each in the following order:

        n = [stress][strain][field]
    
    The information about the parameters for the simulation
    run is stored in the file:

        output / simulation.info # formated as a pickle file

    In order to access all the simulation output for any given 
    configuration, refer to the class MCSimulationParser in this
    same module.

    # SIMULATION PARAMETER SPECIFICATION #

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
        self, system_name, model, supercell, 
        stress=None, strain=None, field=None, 
        output_folder="output" 
        ):

        # TODO PARAMETER FILE, SPECIES, NATS

        """
        
        Sets up everything

        Parameters:
        ----------

        - fdf  (string): common fdf base file for all simulations.

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

        self.name = system_name
        self.model = model
        self.supercell = supercell
        self.sim = CGD_SCUPHandler(self.name, "param_file.xml", cfg.SCUP_EXEC)

        self.output_folder = output_folder # current output folder
        
        if stress is None: # stress vector, optional
            self.stress = [np.zeros(6)]
        else:
            self.stress = [np.array(p, dtype=np.float64) for p in stress]

        if strain is None: # strain vector list, optional
            self.strain = [np.zeros(6)]
        else:
            self.strain = [np.array(s, dtype=np.float64) for s in strain]
        
        if field is None: # electric field vector list, optional
            self.field = [np.zeros(3)]
        else:
            self.field = [np.array(f, dtype=np.float64) for f in field]

        # get the current path
        self.current_path = os.getcwd()

        # create output directory
        try:
            self.main_output_folder = os.path.join(self.current_path, self.output_folder)
            os.makedirs(self.main_output_folder)
            self.DONE = False
        except FileExistsError: # check whether directory already exists
            if cfg.OVERWRITE:
                print("""
                Found already existing output 
                folder named "{}",
                all its contents are now lost.
                Reason: OVERWRITE set to True.""".format(self.output_folder))
                rmtree(self.main_output_folder)
                os.makedirs(self.main_output_folder)
                self.DONE = False
                print("")
                pass
            else:
                print("""
                Found already existing output 
                folder named "{}",
                skipping simulation process.
                Reason: OVERWRITE set to False.""".format(self.output_folder))
                self.SETUP = True
                self.DONE = True
                return 0

        # adjust the supercell as needed
        self.sim.settings["supercell"] = [list(self.supercell)]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # record current settings from ezSCUP.settings
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        
        self.cgd_max_iter = int(cfg.CGD_MAX_ITER)
        self.cgd_force_threshold = float(cfg.CGD_FORCE_THRESHOLD)
        self.cgd_force_factor = float(cfg.CGD_FORCE_FACTOR)
        self.cgd_stress_factor = float(cfg.CGD_STRESS_FACTOR)
        self.cgd_max_step = float(cfg.CGD_MAX_STEP) 
        self.fixed_strain_components = cfg.FIXED_STRAIN_COMPONENTS
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # setup restart file generator
        self.generator = Geometry(self.supercell, model["species"], model["nats"])

        # print common simulation settings
        if cfg.PRINT_CONF_SETTINGS:
            print("Starting FDF settings:")
            self.sim.print()
            print("")

        # save simulation setup file 
        print("Saving simulation setup file... ")

        setup = {
            "run_type": "cgd",

            "name": self.name,
            "model": self.model,
            "supercell": self.supercell,
            "strain": self.strain,
            "stress": self.stress,
            "field": self.field,

            "cgd_max_iter": self.cgd_max_iter,
            "cgd_force_threshold": self.cgd_force_threshold,
            "cgd_force_factor": self.cgd_force_factor,
            "cgd_stress_factor": self.cgd_stress_factor,
            "cgd_max_step": self.cgd_max_step,

            "fixed_strain_components": self.fixed_strain_components,
            }

        simulation_setup_file = os.path.join(self.main_output_folder, cfg.SIMULATION_SETUP_FILE)

        with open(simulation_setup_file, "wb") as f:
            pickle.dump(setup, f)

        print("\nSimulation run has been properly configured.")
        print("You may now proceed to launch it.")

        self.SETUP = True # simulation run setup nicely

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def change_output_folder(self, new_output_folder):

        # get the current path
        self.current_path = os.getcwd()

        previous = self.output_folder

        # create output directory
        print("\nCreating new output folder...")
        try:
            main_output_folder = os.path.join(self.current_path, new_output_folder)
            os.makedirs(main_output_folder)
            self.output_folder = new_output_folder
            self.main_output_folder = main_output_folder
            self.DONE = False

        except FileExistsError: # check whether directory already exists
            if cfg.OVERWRITE:
                print("""
                Found already existing output 
                folder named "{}",
                all its contents are now lost.
                Reason: OVERWRITE set to True.""".format(self.output_folder))
                rmtree(main_output_folder)
                os.makedirs(main_output_folder)
                self.output_folder = new_output_folder
                self.main_output_folder = main_output_folder
                self.DONE = False
                print("")
                pass
            else:
                print("""
                Found already existing output 
                folder named "{}", aborting folder swap.
                Reason: OVERWRITE set to False.""".format(self.output_folder))
                raise ezSCUP.exceptions.PreviouslyUsedOutputFolder()

        # save simulation setup file 
        print("Saving simulation setup file... ")

        setup = {
            "run_type": "cgd",

            "name": self.name,
            "model": self.model,
            "supercell": self.supercell,
            "strain": self.strain,
            "stress": self.stress,
            "field": self.field,

            "cgd_max_iter": self.cgd_max_iter,
            "cgd_force_threshold": self.cgd_force_threshold,
            "cgd_force_factor": self.cgd_force_factor,
            "cgd_stress_factor": self.cgd_stress_factor,
            "cgd_max_step": self.cgd_max_step,

            "fixed_strain_components": self.fixed_strain_components,
            }
        simulation_setup_file = os.path.join(self.main_output_folder, cfg.SIMULATION_SETUP_FILE)

        with open(simulation_setup_file, "wb") as f:
            pickle.dump(setup, f)

        print('\nOutput folder swapped from "{}" to "{}".'.format(previous, self.output_folder))

        pass

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def independent_launch(self, start_geo = None):

        """
        
        Start a simulation run with all the possible combinations 
        of the given parameter grid. 

        Parameters:
        ----------

        - start_geo (Geometry): starting geometry of every simulation.

        """

        # check if setup() has been run
        if self.SETUP != True:
            raise ezSCUP.exceptions.MissingSetup(
            "Run MCSimulation.setup() before launching any simulation."
            )

        # check if somulation has already been carried out
        if self.DONE == True:
            return 0

        print("\n ~ Independent simulation run engaged. ~")

        # checks restart file matches loaded geometry
        if start_geo != None and isinstance(start_geo, Geometry):

            print("\nApplying starting geometry...")
            
            if not np.all(self.generator.supercell == start_geo.supercell): 
                raise ezSCUP.exceptions.GeometryNotMatching()

            if self.generator.nats != None and (start_geo.nats != self.model["nats"]):
                raise ezSCUP.exceptions.GeometryNotMatching()

            if self.generator.species != None and (set(self.model["species"]) != set(self.model["species"])):
                raise ezSCUP.exceptions.GeometryNotMatching()

            self.generator.displacements = start_geo.displacements

        # get a copy of the model file
        copy(self.model["file"], "param_file.xml")

        # parser to get equilibrium geometry
        parser = CGDSimulationParser(output_folder=self.output_folder)

        # simulation counters
        total_counter   =  0

        # total number of simulations 
        nsims = len(self.strain)*len(self.field)*len(self.stress)
        
        # starting time of the simulation process
        main_start_time = time.time()

        print("\nStarting calculations...\n")
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
                    print("Stress:",        str(p),"GPa")
                    print("Strain:",        str(s), r"% change")
                    print("Electric Field:",str(f), "V/m")
                    print("##############################")

                    # file base name
                    sim_name = self.name + "_CGD"
                    self.sim.settings["system_name"].value = sim_name

                    # configuration name
                    conf_name = "c{:02d}{:02d}{:02d}".format(stress_counter, 
                        strain_counter, field_counter)

                    # subfolder name
                    subfolder_name = self.name + "." + conf_name
                        
                    if self.stress != None: # modify target stress, if needed
                        self.sim.settings["external_stress"] = [p]
                    else:
                        p = np.zeros(6)
                    
                    if self.strain != None: # set target strain, if needed
                        self.generator.strains = s
                    else:
                        s = np.zeros(6)

                    if self.field != None: # modify target field, if needed
                        self.sim.settings["static_electric_field"] = [f]
                    else:
                        f = np.zeros(3)

                    # define human output filename
                    output = sim_name + ".out"

                    # create restart file
                    self.sim.settings["geometry_restart"] = FDFSetting(sim_name + ".restart")
                    self.generator.write_restart(sim_name + ".restart")

                    # simulate the current configuration
                    self.sim.launch(output_file=output)

                    # move all the output to its corresponding folder
                    configuration_folder = os.path.join(self.main_output_folder, subfolder_name)
                    os.makedirs(configuration_folder)

                    files = os.listdir(self.current_path)
                    for fi in files:
                        if sim_name in fi:
                            move(fi, configuration_folder)
                    
                    # calculate the energy
                    geo    = parser.access_geometry(p=p, s=s, f=f)
                    energy = SPRun("param_file.xml", geo)
                    with open("energy.pickle", "wb") as f:
                        pickle.dump(energy, f)
                    move("energy.pickle", configuration_folder)

                    # finish time of the current conf
                    conf_finish_time = time.time()

                    conf_time = conf_finish_time-conf_start_time

                    print("\nConfiguration finished! (time elapsed: {:.3f}s)".format(conf_time))
                    print("All files stored in output/" + subfolder_name + " succesfully.\n")

        self.generator.reset_geom()

        # remove the copy of the model
        os.remove("param_file.xml")

        main_finished_time = time.time()
        main_time = main_finished_time - main_start_time

        print("Simulation process complete!")
        print("Total simulation time: {:.3f}s".format(main_time))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def sequential_launch_by_strain(self, start_geo = None, inverse_order = False):

        """
        
        Simulation run where the equilibrium geometry of the simulation for 
        the previous temperature is used as starting geometry of the next one.

        Parameters:
        ----------

        - start_geo (RestartGenerator): starting geometry of the first temperature.

        """

        # check if setup() has been run
        if self.SETUP != True:
            raise ezSCUP.exceptions.MissingSetup(
            "Run MCSimulation.setup() before launching any simulation."
            )

        # check if simulation has already been carried out
        if self.DONE == True:
            return 0

        print("\n ~ Sequential CGD simulation run by strain engaged. ~ ")

        # checks restart file matches loaded geometry
        if start_geo != None and isinstance(start_geo, Geometry):

            print("\nApplying starting geometry...")
            
            if not np.all(self.generator.supercell == start_geo.supercell): 
                raise ezSCUP.exceptions.GeometryNotMatching()

            if self.generator.nats != None and (start_geo.nats != self.model["nats"]):
                raise ezSCUP.exceptions.GeometryNotMatching()

            if self.generator.species != None and (set(self.model["species"]) != set(self.model["species"])):
                raise ezSCUP.exceptions.GeometryNotMatching()

        # get a copy of the model file
        copy(self.model["file"], "param_file.xml")

        # parser to get equilibrium geometry
        parser = CGDSimulationParser(output_folder=self.output_folder)

        # adjust strain ordering
        if inverse_order:
            strain_sequence = list(reversed(list(self.strain)))
        else:
            strain_sequence = list(self.strain)

        # simulation counters
        total_counter   =  0

        # total number of simulations 
        nsims = len(self.strain)*len(self.field)*len(self.stress)
        
        # starting time of the simulation process
        main_start_time = time.time()

        print("\nStarting calculations...\n")
        for p in self.stress:
            stress_counter = [np.array_equal(p,x) for x in self.stress].index(True)
            for f in self.field:
                field_counter = [np.array_equal(f,x) for x in self.field].index(True)
            
                # set starting geometry
                if start_geo != None and isinstance(start_geo, Geometry):
                    self.generator.displacements = start_geo.displacements

                scount = 0
                for s in strain_sequence:
                    strain_counter = [np.array_equal(s,x) for x in self.strain].index(True)
                    
                    # update simulation counter
                    total_counter += 1
                    scount += 1

                    # starting time of the current configuration
                    conf_start_time = time.time()

                    print("##############################")
                    print("Configuration " + str(total_counter) + " out of " + str(nsims))
                    print("Stress:",        str(p),"GPa")
                    print("Strain:",        str(s), r"% change")
                    print("Electric Field:",str(f), "V/m")
                    print("##############################")

                    # file base name
                    sim_name = self.name + "_CGD"
                    self.sim.settings["system_name"].value = sim_name

                    # configuration name
                    conf_name = "c{:02d}{:02d}{:02d}".format(stress_counter, 
                                    strain_counter, field_counter)

                    # subfolder name
                    subfolder_name = self.name + "." + conf_name
                        
                    if self.stress != None: # modify target stress, if needed
                        self.sim.settings["external_stress"] = [p]
                    
                    if self.strain != None: # set target strain, if needed
                        self.generator.strains = s

                    if self.field != None: # modify target field, if needed
                        self.sim.settings["static_electric_field"] = [f]

                    # define human output filename
                    output = sim_name + ".out"

                    # create restart file
                    self.sim.settings["geometry_restart"] = FDFSetting(sim_name + ".restart")
                    self.generator.write_restart(sim_name + ".restart")

                    # simulate the current configuration
                    self.sim.launch(output_file=output)

                    # move all the output to its corresponding folder
                    configuration_folder = os.path.join(self.main_output_folder, subfolder_name)
                    os.makedirs(configuration_folder)

                    files = os.listdir(self.current_path)
                    for fi in files:
                        if sim_name in fi:
                            move(fi, configuration_folder)

                    # calculate the energy
                    geo    = parser.access_geometry(p=p, s=s, f=f)
                    energy = SPRun("param_file.xml", geo)
                    with open("energy.pickle", "wb") as f:
                        pickle.dump(energy, f)
                    move("energy.pickle", configuration_folder)
                    
                    # finish time of the current conf
                    conf_finish_time = time.time()

                    conf_time = conf_finish_time-conf_start_time

                    print("\nConfiguration finished! (time elapsed: {:.3f}s)".format(conf_time))

                    # grab final geometry for next run if needed
                    if scount < len(strain_sequence): 
                        print("Grabbing geometry for the next run...")
                        geo = parser.access_geometry(p=p, s=s, f=f)
                        self.generator.displacements = geo.displacements

                    print("All files stored in output/" + subfolder_name + " succesfully.\n")

        # remove the copy of the model
        os.remove("param_file.xml")

        self.generator.reset_geom()

        main_finished_time = time.time()
        main_time = main_finished_time - main_start_time

        print("Simulation process complete!")
        print("Total simulation time: {:.3f}s".format(main_time))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
