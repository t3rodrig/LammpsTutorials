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

try:
    import pyvoro #pyvoro calls voro++ library, which unlike scipy can calculate cell volumes
except ImportError:
    raise ImportError("Failed to import pyvoro.\n" +
        "install it with: sudo pip install pyvoro\n")


import numpy as np
import scipy as sp
#from scipy.spatial import Voronoi,Delaunay #use pyvoro instead
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg') #don't create a tk window to show figures. Just save them as files.
import matplotlib.pyplot as plt
import copy
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
parser=ap.ArgumentParser(description="Analyzes and plots Volume distribution of water (centerd on oxygen) as function of z-slice.",
                         formatter_class=ap.ArgumentDefaultsHelpFormatter) #better help
parser.add_argument("-p","--topology", type=str,default="../liquid-vapor/data.spce.old.txt",help="Input topology file readable by MDAnalysis.")
parser.add_argument('-t',"--trajectory", type=str,default="traj_centered.dcd",help="Input trajectory file readable by MDAnalysis.")
parser.add_argument('-s',"--skip", type=int,default=1,help="Process every nth frame of the trajectory")
parser.add_argument('-v',"--maxvolume", type=float,default=100.0,help="Maximum expected volume in A^3. The distribution plot will end here.")
parser.add_argument('-dv',"--volumestep", type=float,default=2.0,help="Step size in the volume distribution diagram in A^3.")
args=parser.parse_args()


#load topology and trajectory
u = load_traj(args.topology, args.trajectory)
all_atoms = u.select_atoms("all")


#slow warning
if(len(u.trajectory)/args.skip > 1000):
    print("\nVoronoi tessalation is slow and this will take a long time.")
    print("It is recomended that you run this code with -s %d or higher.\n"%(len(u.trajectory)/1000))


print("\nCalculating molecular properties for each z-slice.")
Nslices = 100
Nbins   = int(args.maxvolume/args.volumestep)+1
sliceW  = u.dimensions[2]/Nslices #in Angstroms
oxygens = u.select_atoms("type 1")

distrib = np.zeros((Nslices, Nbins), dtype=float)


#analyze trajectory
#for ts in u.trajectory[:int(len(u.trajectory)/100):args.skip]: #degug: analyze only the start of trajectory
for ts in u.trajectory[::args.skip]:
    print((CURSOR_UP_ONE + ERASE_LINE),"Processing frame",ts.frame,"of",len(u.trajectory))

    pos=oxygens.get_positions()

    #first, compute voronoi tessalation
    cells = pyvoro.compute_voronoi( pos,
        [[0.0, ts.dimensions[0]], [0.0, ts.dimensions[1]], [0.0, ts.dimensions[2]]], # limits
        2.0, # block size, in A
        periodic=[True]*3 # periodicity
        )

    #loop over atoms
    for i in range(oxygens.n_atoms):
        sliceID = int(pos[i][2]/sliceW)
        
        #next find volumes
        volume = cells[i]['volume']

        #then distribute them to bins
        binID = int(volume/args.volumestep)
        if(binID<Nbins):
            distrib[sliceID, binID] += 1.0


#normalise distributions
for k in range(distrib.shape[0]):
    s=np.sum(distrib[k])
    distrib[k]/=s


#build axes
z    = np.linspace(sliceW*0.5, u.dimensions[2]+sliceW*0.5, num=Nslices, endpoint=False)
vol  = np.linspace(args.volumestep*0.5, args.volumestep*(Nbins+0.5), num=Nbins, endpoint=False)     #in ps
X, Y = np.meshgrid(vol, z)  # `plot_surface` expects `x` and `y` data to be 2D


#plot z_aoutocor
fig1 = plt.figure(figsize=(6,8), dpi=300)
plt.suptitle("Molecular Volume and Coordination Number")
ax1 = fig1.add_subplot(111, projection='3d') #2 rows, 1 column, plot number 1
ax1.set_xlabel(r'Volume($A^3$)')
ax1.set_ylabel(r'Z-slice ($A$)')
ax1.set_zlabel(r'Normalized Volume Distribution')
ax1.plot_wireframe(X,Y, distrib)

###plot xy_aoutocor
##ax2 = fig1.add_subplot(212, projection='3d') #2 rows, 1 column, plot number 1
##ax2.set_xlabel(r'Number of Neigbours')
##ax2.set_ylabel(r'Z-slice ($A$)')
##ax2.set_zlabel(r'Normalized Coordination Number Distribution')
##ax2.plot_surface(X,Y, distrib) #WARNING Coordination Number NOT IMPLEMENTED!

#plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0, rect=(0,0,1,0.95))
fig1.savefig("volume_z_slice.png")
plt.close(fig1)


exit()
