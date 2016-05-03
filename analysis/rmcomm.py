#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import sys

# Check correct version of python
if sys.version >= '3':
    raise Exception("Error: MDAnalysis can't run right now under Python 3 \n" +
    "https://github.com/MDAnalysis/mdanalysis/wiki/GSoC-2016-Project-Ideas \n")

try:
    import MDAnalysis
except ImportError:
    raise ImportError("Detailed information on how to install MDAnalysis " +
        " can be found on the official website:\n" +
        "https://github.com/MDAnalysis/mdanalysis/wiki/Install \n" )

import numpy as np
import scipy as sp
import argparse as ap

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


def load_traj(top_file, traj_file):
    """
    Load trajectory from a file and return it as instance of MDAnalysis

    Parameters
    ----------
    traj_file : str
                trajectory file name (traj.xtc, trj.dcd, trj.xyz)
    top_file : str
                topology file name (data.topol.txt)

    Returns
    -------
    u : MDAnalysis.Universe
    """
    try:
        u = MDAnalysis.Universe(top_file, traj_file, topology_format="DATA",
                                format="LAMMPS")
    except:
        raise IOError("No such file or directory: " + top_file + " or " + traj_file)
    return u


###############
#Entry point:
###############

#parse arguments
parser=ap.ArgumentParser(description="Removes center of mass motion form trajectory by centering each frame in the simulation box.",
                         formatter_class=ap.ArgumentDefaultsHelpFormatter) #better help
parser.add_argument("-p","--topology", type=str,default="../liquid-vapor/data.spce.old.txt",help="Input topology file readable by MDAnalysis.")
parser.add_argument('-t',"--trajectory", type=str,default="../liquid-vapor/traj.dcd",help="Input trajectory file readable by MDAnalysis.")
parser.add_argument('-o',"--output", type=str,default="traj_centered.dcd",help="Output trajectory file writable by MDAnalysis.")
parser.add_argument('-s',"--skip", type=int,default=1,help="Output every nth frame of the trajectory")
args=parser.parse_args()



#load topology and trajectory
u = load_traj(args.topology, args.trajectory)
all_atoms = u.select_atoms("all")

#open output file
with MDAnalysis.Writer(args.output, all_atoms.n_atoms) as W:

    print("Removing the COM motion:\n")    
    #itterate through frames
    for ts in u.trajectory:
        if(ts.frame%args.skip!=0): continue     #skip?
        print((CURSOR_UP_ONE + ERASE_LINE),"Processing frame",ts.frame,"of",len(u.trajectory))
        com=all_atoms.center_of_mass(pbc=True)  #wrap all atoms into box and return the center of mass
        shift=u.dimensions[:3]*0.5 - com
        coords=all_atoms.coordinates()          #coordinates of the atoms, passed by reference
        coords+=shift                           #center com in box so that liquid doesn't drift on z-axis
        all_atoms.pack_into_box()               #rewrap if any atoms moved outside box beacuse of centering

        W.write(all_atoms)                      #write output
        

exit()
