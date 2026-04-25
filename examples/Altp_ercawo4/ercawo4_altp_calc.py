#!/usr/bin/env python3
"""
Calculate Altp (coordination) parameters for Er:CaWO4 from crystal structure.

This example demonstrates:
1. Loading crystal structure from CIF file (Materials Project)
2. Extracting atomic coordinates from pymatgen
3. Calculating spherical polar coordinates of ligands
4. Computing Altp coordination parameters for crystal field calculations
5. Interfacing crystal structure data with pycf calculations

The example calculates Er^3+ coordination geometry in CaWO4 host,
extracting parameters needed for crystal field Hamiltonian construction.

Prerequisites:
- pymatgen (for CIF parsing)
- Crystal structure CIF file (e.g., from Materials Project)
- numpy, pycf.paramcalc
"""

import numpy as np
import numpy.linalg as LA
from pymatgen.io.cif import CifParser
from pycf.paramcalc import *    # moved these functions from the old inten.py
from pycf.cfl_util import *

def nn_coords(struct, site, r): 
    """
    Return the spherical polar coordinates of the next nearest neighbors within radius r.
    """          
    NN = struct.get_neighbors(site, r)
    origin = site.coords
    
    spc = np.zeros([len(NN), 3])
    for i,N in enumerate(NN):
        xyz = N[0].coords-origin
        spc[i, 0] = LA.norm(xyz)
        spc[i, 1] = np.arccos(xyz[2]/spc[i,0])
        spc[i, 2] = np.arctan2(xyz[1],xyz[0])

    return spc

# cif file from https://materialsproject.org/materials/mp-19426/
parser = CifParser("CaWO4_mp-19426_conventional_standard.cif")
cawo4_struct = parser.get_structures()[0]

# Add nearest neighbors in 3 Angstrom radius to summary string, which is all the
# oxygens.
Ca1 = cawo4_struct.sites[0]
nn = cawo4_struct.get_neighbors(Ca1, 3.0)
s = ""
heading = "Nearest neighbors\n"
s += uline_char(heading)
for n in nn:
    s += "Species = %s, r = %f Angstrom\n" % (n[0].species.chemical_system, Ca1.distance(n[0]))
s += "\n"

# Get coordinates of all ligands within 3 Angstrom.
nn_ligands = nn_coords(cawo4_struct, Ca1, 3)

# To calc Altp we also need the charge for each ligand (-2 for Oxygen), and mean
# polarizability (-3.2).
ligands = [Ligand(c, -2, -3.2) for c in nn_ligands]

# Lanthanide dopant type and charge, along with list of Ligand objects.
Altp = AltpData('Er', 3, ligands)
A_list= Altp.eval_params()

s += Altp.gen_summary()
print(s)
