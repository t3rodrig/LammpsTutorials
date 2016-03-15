#!/usr/bin/env python

import numpy as np
import scipy as sp
import MDAnalysis
import matplotlib.pyplot as plt



###############
#Entry point:
###############


#load topology and trajectory
topFname="../liquid-vapor/data.spce.old.txt"    #topology_format="DATA"
trajFname="../liquid-vapor/traj.dcd"            #format="LAMMPS"
u = MDAnalysis.Universe(topFname, trajFname, topology_format="DATA", format="LAMMPS")
all_atoms = u.select_atoms("all")

mu_history=[]
cell_history=[]
#itterate through frames
for ts in u.trajectory:
    #calculate the total dipole moment of the simulation cell
    acoords=all_atoms.pack_into_box(inplace=False)  #make sure atoms are in first periodic cell
    temp=np.multiply(all_atoms.charges[:,np.newaxis], acoords) #Q*r
    mu=np.sum(temp, axis=0) #sum(Q*r)
    muMag=np.linalg.norm(mu)/0.20819434 #magnitude in Debye
    mu_history.append(muMag)
    cell_history.append(np.array(u.dimensions))   #keep track of dimentions of the periodic cell

#build histogram
mu_history=np.array(mu_history)
max_mu=np.amax(mu_history)
min_mu=np.amin(mu_history)
avg=np.mean(mu_history)
print "<mu>=",avg,"Debye"
print "<mu^2>=",np.mean(mu_history**2),"Debye"


step = 10    #spacing between points on the histogram in Debye
#shift max and min so that ends of the histogram are zero
min_mu-=2*step
max_mu+=2*step

histogram=np.zeros(int((max_mu-min_mu)/step)) #allocate space
x=np.linspace(min_mu, min_mu+step*histogram.shape[0], histogram.shape[0])
for mu in mu_history:
    i=int((mu-min_mu)/step)
    histogram[i]+=1         #increment count for entries in this region

#normalize the distribution
histogram/=histogram.shape[0]
    

#draw and save plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Total Dipole Moment (Debye)')
ax.set_ylabel('Distribution')
ax.text((min_mu+max_mu)/2, 0.04, r'$<\mu>=%.3f$ Debye'%avg+'\n'+'$<\mu^2>=%.3f$ Debye'%np.mean(mu_history**2))
ax.plot(x, histogram, '-')

fig.savefig("dipole.png")



#gather data about diole dependence on cell length
#assume isotropic barostat (all directions scale proportionally)
#this way we only need to look at one cell edge
cell_history=np.vstack(cell_history)
Lstep=0.2
min_L=np.amin(cell_history[:,0]) - 2*Lstep
max_L=np.amax(cell_history[:,0]) + 2*Lstep

avgMu=np.zeros(int((max_L-min_L)/Lstep)) #allocate space
count=np.zeros(int((max_L-min_L)/Lstep), dtype=int)
L=np.linspace(min_L, min_L+Lstep*avgMu.shape[0], avgMu.shape[0])
for i in range(cell_history.shape[0]):
    j=int((cell_history[i][0]-min_L)/Lstep)  #where in the arrays do we write to
    count[j]+=1         #increment count for entries in this region
    avgMu[j]+=mu_history[i]

#compute the average mu for each group of side lengths
#do so while gracefully handling divide by 0
np.seterr(invalid='ignore')
avgMu/=count 
np.seterr(invalid='warn')
avgMu=np.where(np.isfinite(avgMu), avgMu, 0) #replace +-inf and nan with 0
   

#draw and save plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(u'Length of Cell Side (\u212B)')
ax.set_ylabel('Averaged Dipole Moment of Simulation Cell (Debye)')
ax.plot(L, avgMu, '-')

fig.savefig("dipole_vs_L.png")

exit()
