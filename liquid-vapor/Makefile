# Detailed information on how to build LAMMPS can be found on the official
# website (http://lammps.sandia.gov)
LAMMPS=~/bin/lmp

traj.lammpstrj : in.spce.txt data.spce.old.txt
	mpiexec -np 8 $(LAMMPS) -sc log.sc.txt -in $< 

.PHONY : clean
clean :
	rm -rf log.*
	rm -rf data.spce.new.txt rst.spce.new.bin
	rm -rf traj.* confout.lammpstrj
