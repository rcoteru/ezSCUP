# ezSCUP: SCALE-UP made easy

ezSCUP is a Python module designed to facilitate the execution of SCALE-UP simulations in a parameter grid of temperatures, pressures, strains and/or electric fields. It provides a high-level interface to handle the program and its output, includes several analysis functions concerning structural modes and macroscopic polarization in perovskites. 

SCALE-UP (short for **S**econd-principles **C**omputational **A**pproach for **L**attice and **E**lectrons) is a Second-Principles DFT simulation program written in Fortran and developed by researchers from the University of Cantabria (UC, Spain) and the Luxembourg Institute of Science and Technology (LIST), as well as by a number of collaborators [[+info]](https://www.secondprinciples.unican.es/).  

This module has been developed as an integral part of my Bachelor's final-year thesis on structural transititions within strontium titanate (SrTiO3), a perovskite oxide. Therefore, while the base package contains most of the generic functionality, everything within the submodule *[ezSCUP/srtio3](ezSCUP/srtio3)* is designed specifically towards this material.


## Features

- Perform single-point and geometry optimization (CGD) calculations.
- Schedule Monte Carlo simulation runs in a grid of temperatures, strains, pressures and electric fields.
- Set starting geometries for each simulation run: simulate everything sequentially or independently.
- Easy output access and management for each configuration through Python data structures.
- Generates an equilibrium geometry for each Monte Carlo configuration.
- Obtain the normal vibrational modes of a material through Hessian diagonalization. 
- Projection of the resulting structural modes (i.e. AFD or FE modes in SrTiO3) onto any geometry.
- Calculation of macroscopic polarization through the use of Born effective charges.
- Observe domain structures through a per-unit-cell application of the previous features.

## Installation

Although the package may be installed with a simple command (```pip3 install ezSCUP```), it is advisable to download the source code and place the *"ezSCUP"* folder in your working directory, so as to always have the most up-to-date version of the code and an easier access to it in case any modifications are needed.  

## Usage

The source code itself is heavily documented, so as to be almost self explanatory. Working examples may be found in the *"examples"* folder, corresponding to several use cases explored in my thesis. The simulation settings (number of MC steps, supercell size) have been tuned down in order to fit the whole simulation run in a few hours, as opposed to the several weeks it took in the original work. Explanations on each different setup may be found within the corresponding folder, although I must note that the SrTiO3 model used in these examples, as well as in the *"tests"* folder, corresponds to the one used by Wojdeł et al. in [this article](https://iopscience.iop.org/article/10.1088/0953-8984/25/30/305401).

<p align="center"> 
<img src="example.png">
</p>
<p align="center"> 
Representation of the antiferrodistortive phase transition present in bulk SrTiO3 simulated with ezSCUP, <br> using 8x8x8 and 10x10x10 supercells with 40,000-80,000 MC steps per simulation.
</p>
