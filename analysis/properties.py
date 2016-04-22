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
matplotlib.use('Agg') #don't create a tk window to show figures. Just save them as files.
import matplotlib.pyplot as plt
import copy

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

#load topology and trajectory
topFname="data.spce.old.txt"    #topology_format="DATA"
trajFname="traj.dcd"            #format="LAMMPS"
u = load_traj(topFname, trajFname)

all_atoms = u.select_atoms("all")

print("Calculating properties of the whole simulation cell:")

mu_history=[]
sumV=0
#itterate through frames
for ts in u.trajectory:
    #calculate the total dipole moment of the simulation cell
    
    #com=all_atoms.center_of_mass(pbc=True)  #wrap all atoms into box and return the center of mass
    #shift=u.dimensions[:3]*0.5 - com
    coords=all_atoms.coordinates()          #coordinates of the atoms, passed by reference
    #coords+=shift                           #center com in box so that liquid doesn't drift on z-axis
    #all_atoms.pack_into_box()               #rewrap if any atoms moved outside box beacuse of centering
        
    temp=np.multiply(all_atoms.charges[:,np.newaxis], coords) #Q*r
    mu=np.sum(temp, axis=0) #sum(Q*r)
    muMag=np.linalg.norm(mu)/0.20819434 #magnitude in Debye
    mu_history.append(muMag)

    sumV+=ts.volume

#build histogram
mu_history=np.array(mu_history)
max_mu=np.amax(mu_history)
min_mu=np.amin(mu_history)
avg=np.mean(mu_history)
print("\tTotal Cell dipole moment:")
print("<mu> =",avg,"Debye")
print("<mu^2> =",np.mean(mu_history**2),"Debye")

#find dielectric constant
#this is from: http://www.tandfonline.com/doi/abs/10.1080/00268978300102721
#as implemented in Gromacs 4.7.?
temp=300 #K
eps0=8.854187817e-12 # epsilon_0 in C^2 J^-1 m^-1
volume=sumV*1e-30/mu_history.shape[0] #in m^3
kb = 1.38064852e-23  # Boltzmann constant in m^2 kg/(s^2 K)
fac= 1.112650021e-59 # converts Debye^2 to C^2 m^2
epsilon=1+((avg**2 - np.mean(mu_history**2))*fac/(3.0*eps0*volume*kb*temp))
print("epsilon =",np.mean(mu_history**2)," assuming temp =",temp,"K")


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
plt.ioff()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Total Dipole Moment (Debye)')
ax.set_ylabel('Number of Occurances')
ax.plot(x, histogram, '-')

fig.savefig("dipole.png")
plt.close(fig)



#Calculate avg molecular dipole moments as a function of the z-slice.
#We need groups of individual molecules for this
print("\nCalculating molecular properties for each z-slice.")
Nslices=50
Nframes=0
sliceW= u.dimensions[2]/Nslices #in Angstroms
W_res = all_atoms.residues #list of residue IDs
slice_mu=np.zeros(Nslices)
slice_mu_z=np.zeros(Nslices)
slice_mol_count=np.zeros(Nslices, dtype=int)

#precache molecules. This is slow and we don't want to do it every time step
mols=[]
for res in W_res:
    mols.append(all_atoms.select_atoms("resid %d"%res.id))
print("There are", len(W_res), "molecules.\n")
    
#analyse trajectory
for ts in u.trajectory:
    #if(ts.frame%100!=0): continue  #use for debug to make code skip frames
    Nframes+= 1
    print((CURSOR_UP_ONE + ERASE_LINE),"Processing frame",ts.frame,"of",len(u.trajectory))
    
    #for molecular dipole moment, molecules have to be wrapped so they stay together
    all_atoms.wrap(compound='residues', center='com')
    #TODO: make center of mass stay at center of box to avoid drift on z-axis
    
    for grp in mols:
        com = grp.center_of_mass() #this determined which slice the molecule is in
        sliceID=int(com[2]/sliceW)
        
        #compute molecular dipole in Debye
        mol_mu=np.sum(np.multiply(grp.charges[:,np.newaxis], grp.coordinates()), axis=0)/0.20819434
        #deep copy to avoid passing by reference and data being overwritten 1 line later
        mol_mu_z=copy.deepcopy(mol_mu[2])#z-component of the molecular dipole
        mol_mu=np.linalg.norm(mol_mu)    #magnitude

        #tabulate
        slice_mu[sliceID]+=mol_mu
        slice_mu_z[sliceID]+=mol_mu_z
        slice_mol_count[sliceID]+=1

#average, while gracefully handling division by 0
with np.errstate(divide='ignore', invalid='ignore'):
        slice_mu = np.true_divide( slice_mu, slice_mol_count )
        slice_mu[ ~ np.isfinite( slice_mu )] = 0  # -inf inf NaN
        slice_mu_z = np.true_divide( slice_mu_z, slice_mol_count )
        slice_mu_z[ ~ np.isfinite( slice_mu_z )] = 0  # -inf inf NaN

#z-coordinates of the centers of the z-slices
z=np.linspace(sliceW*0.5, u.dimensions[2]+sliceW*0.5, num=Nslices, endpoint=False)

#plot dipole_z_slice
fig2 = plt.figure(figsize=(6,8), dpi=300)
plt.suptitle("Molecular Dipole Moment Properties at Different Z-slices")
ax1 = fig2.add_subplot(2,1,1) #2 rows, 1 column, plot number 1
ax1.set_xlabel('Z-slice (A)')
ax1.set_ylabel(r'<|$\mu_{mol}$|> (Debye)')
ax1.plot(z, slice_mu, '-')

ax2 = fig2.add_subplot(2,1,2) #2 rows, 1 column, plot number 2
ax2.set_xlabel('Z-slice (A)')
ax2.set_ylabel(r'<$\mu_{z \, mol}$> (Debye)')
ax2.plot(z, slice_mu_z, '-')

plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0, rect=(0,0,1,0.95))
fig2.savefig("dipole_z_slice.png")
plt.close(fig2)

#plot density_z_slice
fig3 = plt.figure(figsize=(6,8), dpi=300)
plt.suptitle("Density at Different Z-slices")
ax1 = fig3.add_subplot(211)
ax1.set_xlabel('Z-slice (A)')
ax1.set_ylabel('<Number of Molecules>')
avg_slice_mol_count=slice_mol_count/Nframes
ax1.plot(z, avg_slice_mol_count, '-')

ax2 = fig3.add_subplot(2,1,2) #2 rows, 1 column, plot number 2
ax2.set_xlabel('Z-slice (A)')
ax2.set_ylabel(r'<Density> (g/cm${}^3$)')
volume*=1e+6    #volume is in m^3, need it in cm ^3
rho=(avg_slice_mol_count*18.01528/6.0221409e+23)/(volume/(Nslices)) #density in g/cm^3
ax2.plot(z, rho, '-')

plt.tight_layout(pad=1.0, w_pad=3.0, h_pad=1.0, rect=(0,0,1,0.95))
fig3.savefig("density_z_slice.png")
plt.close(fig)

exit()
