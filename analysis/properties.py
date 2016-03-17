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
import matplotlib.pyplot as plt

###############
#Entry point:
###############


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

#load topology and trajectory
topFname="../liquid-vapor/data.spce.old.txt"    #topology_format="DATA"
trajFname="../liquid-vapor/traj.dcd"            #format="LAMMPS"
u = load_traj(topFname, trajFname)

all_atoms = u.select_atoms("all")

mu_history=[]
#itterate through frames
for ts in u.trajectory:
    #calculate the total dipole moment of the simulation cell
    
    com=all_atoms.center_of_mass(pbc=true)  #wrap all atoms into box and return the center of mass
    shift=u.dimensions*0.5 - com
    coords=all_atoms.coordinates()          #coordinates of the atoms, passed by reference
    coords+=shift                           #center com in box so that liquid doesn't drift on z-axis
    all_atoms.pack_into_box()               #rewrap if any atoms moved outside box beacuse of centering
        
    temp=np.multiply(all_atoms.charges[:,np.newaxis], coords) #Q*r
    mu=np.sum(temp, axis=0) #sum(Q*r)
    muMag=np.linalg.norm(mu)/0.20819434 #magnitude in Debye
    mu_history.append(muMag)

#build histogram
mu_history=np.array(mu_history)
max_mu=np.amax(mu_history)
min_mu=np.amin(mu_history)
avg=np.mean(mu_history)
print("<mu>=",avg,"Debye")
print("<mu^2>=",np.mean(mu_history**2),"Debye")


step = 10    #spacing between points on the histogram in Debye
#shift max and min so that ends of the histogram are zero
min_mu-=2*step
max_mu+=2*step

histogram=np.zeros(int((max_mu-min_mu)/step)+1) #allocate space
x=np.linspace(min_mu, min_mu+step*histogram.shape[0], histogram.shape[0])
for mu in mu_history:
    i=int((mu-min_mu)/step)
    histogram[i]+=1         #increment count for entries in this region

#TODO: normalize the distribution


#draw and save plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Total Dipole Moment (Debye)')
ax.set_ylabel('Number of Occurances')
ax.plot(x, histogram, '-')

fig.savefig("dipole.png")

exit()
