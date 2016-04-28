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
parser=ap.ArgumentParser(description="Analyzes and plots a number of properties as a function of z-slice.",
                         formatter_class=ap.ArgumentDefaultsHelpFormatter) #better help
parser.add_argument("-p","--topology", type=str,default="../liquid-vapor/data.spce.old.txt",help="Input topology file readable by MDAnalysis.")
parser.add_argument('-t',"--trajectory", type=str,default="traj_centered.dcd",help="Input trajectory file readable by MDAnalysis.")
parser.add_argument('-s',"--skip", type=int,default=1,help="Process every nth frame of the trajectory")
parser.add_argument('-dt',"--timestep", type=float,default=0.002,help="Timestep in ps.")
parser.add_argument('-l',"--autocorlen", type=float,default=250,help="Length of autocorelations in frames. Reference frames are taken every this many frames.")
args=parser.parse_args()


#load topology and trajectory
u = load_traj(args.topology, args.trajectory)

all_atoms = u.select_atoms("all")


print("\nCalculating molecular properties for each z-slice.")
Nslices=100
sliceW = u.dimensions[2]/Nslices #in Angstroms
oxygens=u.select_atoms("type 1")

z_autocor=np.zeros((Nslices, args.autocorlen), dtype=float)
xy_autocor=np.zeros((Nslices, args.autocorlen), dtype=float)
N_in_slice=np.zeros((Nslices, args.autocorlen), dtype=int)

counts=np.zeros(Nslices, dtype=int)
counts=np.zeros(Nslices, dtype=int)


#Calculate autocorelations for individual oxygens.
#Then add it to whatever slice the oxygen is in in that frame and average.

first=True
vel=0 #declare vel up here, don't want the garbage collector touching it
lastpos=0
refvel=0
k=0
#analyze trajectory
#for ts in u.trajectory[:int(len(u.trajectory)/100):args.skip]: #degug: analyze only the start of trajectory
for ts in u.trajectory[::args.skip]:
    print((CURSOR_UP_ONE + ERASE_LINE),"Processing frame",ts.frame,"of",len(u.trajectory))

    pos=oxygens.get_positions()
    if(first):
        first=False
        lastpos=copy.deepcopy(pos)
        continue

    dif = (lastpos-pos)
    #check for crossing of pbc, if true, dif=(box_dim-|dif|) in direction opposite to dif
    dif[:,0]=np.where(np.abs(dif[:,0])>ts.dimensions[0]*0.5, (-np.sign(dif[:,0]))*(ts.dimensions[0]-np.abs(dif[:,0])), dif[:,0])
    dif[:,1]=np.where(np.abs(dif[:,1])>ts.dimensions[1]*0.5, (-np.sign(dif[:,1]))*(ts.dimensions[1]-np.abs(dif[:,1])), dif[:,1])
    dif[:,2]=np.where(np.abs(dif[:,2])>ts.dimensions[2]*0.5, (-np.sign(dif[:,2]))*(ts.dimensions[2]-np.abs(dif[:,2])), dif[:,2])
    vel = dif/(args.timestep*args.skip) #in A/ps
    
    
    lastpos=copy.deepcopy(pos)

    if((ts.frame-(1*args.skip))%(args.autocorlen*args.skip)==0): #this is a reference frame
        refvel=vel
        k=0     #reset frames since reference

    #compute v(0)*v(t)
    z_corr=vel[:,2]*refvel[:,2]
    ##xy_corr=np.tensordot(vel[:,:2],refvel[:,:2],([1],[1]))
    xy_corr = (vel[:,0]*refvel[:,0]) + (vel[:,1]*refvel[:,1])

    #sliceID=pos[:,2]/sliceW
    for i in range(oxygens.n_atoms):
        sliceID=int(pos[i][2]/sliceW)
        z_autocor[sliceID, k]  +=  z_corr[i]
        xy_autocor[sliceID, k] += xy_corr[i]
        N_in_slice[sliceID, k] += 1

    k+=1        #increment frames since reference


#average, while gracefully handling division by 0
with np.errstate(divide='ignore', invalid='ignore'):
        z_autocor = np.true_divide( z_autocor, N_in_slice )
        z_autocor[ ~ np.isfinite( z_autocor )] = 0  # -inf inf NaN
        xy_autocor = np.true_divide( xy_autocor, N_in_slice )
        xy_autocor[ ~ np.isfinite( xy_autocor )] = 0  # -inf inf NaN

#z-coordinates of the centers of the z-slices and time
z    = np.linspace(sliceW*0.5, u.dimensions[2]+sliceW*0.5, num=Nslices, endpoint=False)
time = np.arange(0, args.autocorlen)*args.timestep     #in ps
X, Y = np.meshgrid(time, z)  # `plot_surface` expects `x` and `y` data to be 2D


#plot z_aoutocor
fig2 = plt.figure(figsize=(6,8), dpi=300)
plt.suptitle("Velocity Autocorrelation")
ax1 = fig2.add_subplot(211, projection='3d') #2 rows, 1 column, plot number 1
ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('Z-slice (A)')
ax1.set_zlabel(r'Velocity Autocorrelation Z-component ($A^2/ps^2$)')
ax1.set_zlim(-5, 20)
ax1.plot_wireframe(X,Y, z_autocor)

#plot xy_aoutocor
ax2 = fig2.add_subplot(212, projection='3d') #2 rows, 1 column, plot number 1
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Z-slice (A)')
ax2.set_zlabel(r'Velocity Autocorrelation XY-component ($A^2/ps^2$)')
ax2.set_zlim(-10, 40)
ax2.plot_surface(X,Y, xy_autocor)

#plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0, rect=(0,0,1,0.95))
fig2.savefig("vel_autocorr_z_slice.png")
plt.close(fig2)



#Diffusion coefficient is the integral of velocity autocorrelation
z_diff_coef  = np.sum( z_autocor, axis=1)
xy_diff_coef = np.sum(xy_autocor, axis=1)/2

fig3 = plt.figure(figsize=(6,8), dpi=300)
plt.suptitle("Diffusion Coefficient")
ax1 = fig3.add_subplot(211) #2 rows, 1 column, plot number 1
ax1.set_xlabel('Z-slice (A)')
ax1.set_ylabel(r'$D_\bot$ ($\mathrm{A}^2/\mathrm{ps}$)')
ax1.plot(z, z_diff_coef, '-')

ax2 = fig3.add_subplot(212) #2 rows, 1 column, plot number 1
ax2.set_xlabel('Z-slice (A)')
ax2.set_ylabel(r'$D_\parallel$ ($\mathrm{A}^2/\mathrm{ps}$)')
ax2.plot(z, xy_diff_coef, '-')

plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0, rect=(0,0,1,0.95))
fig3.savefig("diffusion_z_slice.png")
plt.close(fig3)
exit()
