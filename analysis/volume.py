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
parser.add_argument('-v',"--maxvolume", type=float,default=100.0,help="Maximum expected volume in A^3. The volume distribution plot will end here.")
parser.add_argument('-n',"--maxneighbours", type=int,default=50,help="Maximum expected number of neighbours. The coordination number distribution plot will end here.")
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

v_distrib = np.zeros((Nslices, Nbins), dtype=float)
f_distrib = np.zeros((Nslices, args.maxneighbours+1), dtype=float)


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
        
        #next find volumes and faces
        volume = cells[i]['volume']
        faces  = len(cells[i]['faces'])

        #then distribute them to bins
        binID = int(volume/args.volumestep)
        if(binID<Nbins):
            v_distrib[sliceID, binID] += 1.0
        if(faces<=args.maxneighbours):
            f_distrib[sliceID, faces] += 1.0

#normalise distributions
with np.errstate(divide='ignore', invalid='ignore'):
    for k in range(v_distrib.shape[0]):
        s=np.sum(v_distrib[k])
        v_distrib[k]/=s
        v_distrib[ ~ np.isfinite( v_distrib )] = 0  # -inf inf NaN
        
        s=np.sum(f_distrib[k])
        f_distrib[k]/=s
        f_distrib[ ~ np.isfinite( f_distrib )] = 0  # -inf inf NaN

#build axes
z    = np.linspace(sliceW*0.5, u.dimensions[2]+sliceW*0.5, num=Nslices, endpoint=False)
vol  = np.linspace(args.volumestep*0.5, args.volumestep*(Nbins+0.5), num=Nbins, endpoint=False)     #in ps
X, Y = np.meshgrid(vol, z)  # `plot_surface` expects `x` and `y` data to be 2D

#set up a custom colormap, so that 0 is white, not light blue as with cmap='Blues'
from matplotlib.colors import LinearSegmentedColormap
cdict = {'red':   ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0))}
cm = LinearSegmentedColormap('WhiteBlue', cdict)


#plot z_aoutocor
fig1 = plt.figure(figsize=(12,8), dpi=300)
plt.suptitle("Molecular Volume and Coordination Number")
ax1 = fig1.add_subplot(221, projection='3d') #2 rows, 2 column, plot number 1
ax1.set_xlabel(r'Volume($A^3$)')
ax1.set_ylabel(r'Z-slice ($A$)')
ax1.set_zlabel(r'Normalized Volume Distribution')
ax1.plot_wireframe(X,Y, v_distrib)

ax3 = fig1.add_subplot(223) #2 rows, 2 column, plot number 3
ax3.set_xlabel(r'Volume($A^3$)')
ax3.set_ylabel(r'Z-slice ($A$)')
ax3.set_xlim(0, args.volumestep*Nbins)
ax3.set_ylim(0, u.dimensions[2]+sliceW*0.5)
#ax3.set_zlabel(r'Normalized Volume Distribution')
mapable=ax3.pcolor(X,Y, v_distrib, cmap=cm, vmin=0, vmax=0.5)
plt.colorbar(mapable, ax=ax3)

#plot xy_aoutocor
ax2 = fig1.add_subplot(222, projection='3d') #2 rows, 2 column, plot number 2
ax2.set_xlabel(r'Number of Neigbours')
ax2.set_ylabel(r'Z-slice ($A$)')
ax2.set_zlabel(r'Normalized Coordination Number Distribution')
F    = np.arange(args.maxneighbours+1)
X, Y = np.meshgrid(F, z)
ax2.plot_wireframe(X,Y, f_distrib)

ax4 = fig1.add_subplot(224) #2 rows, 2 column, plot number 4
ax4.set_xlabel(r'Number of Neigbours')
ax4.set_ylabel(r'Z-slice ($A$)')
ax4.set_xlim(0, args.maxneighbours)
ax4.set_ylim(0, u.dimensions[2]+sliceW*0.5)
#ax4.set_zlabel(r'Normalized Coordination Number Distribution')
mapable=ax4.pcolor(X,Y, f_distrib, cmap=cm, vmin=0, vmax=0.5)
plt.colorbar(mapable, ax=ax4)

#format and save to file
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0, rect=(0,0,1,0.95))
fig1.savefig("volume_z_slice.png")
plt.close(fig1)


exit()
