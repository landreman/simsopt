#!/usr/bin/env python

import numpy as np

from simsopt.util.mpi import log, MpiPartition
from simsopt.mhd import Vmec, Spec, Boozer, Quasisymmetry
from simsopt.mhd.spec import Residue
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve

"""
In this example, we simultaneously optimize for quasisymmetry and
the elimination of magnetic islands, with both VMEC and SPEC called in
the objective function.
"""

log()
mpi = MpiPartition()
mpi.write()

vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA_iota0.4_withIslands'), mpi=mpi)
surf = vmec.boundary

spec = Spec(os.path.join(os.path.dirname(__file__), 'inputs', 'nfp2_QA_iota0.4_withIslands.sp'), mpi=mpi)

# This next line is where the boundary surface objects of VMEC and
# SPEC are linked:
spec.boundary = surf

# Define parameter space:
surf.all_fixed()
surf.fixed_range(mmin=0, mmax=3,
                 nmin=-3, nmax=3, fixed=False)
surf.set_fixed("rc(0,0)")  # Major radius

# Configure quasisymmetry objective:
qs = Quasisymmetry(Boozer(vmec),
                   0.5,  # Radius to target
                   1, 0)  # (M, N) you want in |B|

# iota = p / q
p = -2
q = 5
residue1 = Residue(spec, p, q)
residue2 = Residue(spec, p, q, theta=np.pi)

if mpi.group == 0:
    r1 = residue1.J()
    r2 = residue2.J()
if mpi.proc0_world:
    print("Initial residues:", r1, r2)
#exit(0)

# Define objective function
prob = LeastSquaresProblem([(vmec.aspect, 6, 1),
                            (vmec.iota_axis, 0.385, 1),
                            (vmec.iota_edge, 0.415, 1),
                            (qs, 0, 1),
                            (residue1, 0, 2),
                            (residue2, 0, 2)])

least_squares_mpi_solve(prob, mpi=mpi, grad=True)

if mpi.group == 0:
    r1 = residue1.J()
    r2 = residue2.J()
if mpi.proc0_world:
    print("Final residues:", r1, r2)

print("Good bye")
