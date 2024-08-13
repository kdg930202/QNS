import netket as nk
from numpy.lib.function_base import append
from itertools import permutations, combinations
from netket.vqs.mc import get_local_kernel_arguments, get_local_kernel
import time
import matplotlib.pyplot as plt
import json
import numpy as np
from numpy.linalg import eig
from flax.core.frozen_dict import FrozenDict
import flax
from netket.optimizer.qgt import QGTJacobianPyTree


from typing import Callable, Tuple
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
from netket.utils.types import PyTree
from netket.operator.spin import sigmax, sigmay,sigmaz

import matplotlib.pyplot as plt
from tqdm import tqdm
###################################################################
from scipy.sparse.linalg import eigsh
###################################################################
import os
import pandas as pd


###
def alpha_depen(learning_rate, n_steps, alpha_, hi, g, ha, Et):


  ma = nk.models.RBM(alpha=alpha_, param_dtype=complex)


  # Build the sampler
  sa = nk.sampler.MetropolisExchange(hilbert=hi,graph=g)

  vs = nk.vqs.FullSumState(hi, ma)
  # vs = nk.vqs.MCState(sa, ma, n_samples=1008)
  # holo_check = nk.utils.is_probably_holomorphic(vs._apply_fun, vs.parameters, vs.samples, vs.model_state)
  # print(holo_check)
  # vs = vs.quantum_geometric_tensor(QGTJacobianPyTree(holomorphic=True))

  # Optimizer
  op = nk.optimizer.Sgd(learning_rate=learning_rate)

  # Stochastic Reconfiguration
  sr = nk.optimizer.SR(diag_shift=0.0001, holomorphic=True)
  # sr = nk.optimizer.SR(solver=nk.optimizer.solver.cholesky)
  # qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
  # holomorphic – boolean indicating if the ansatz is boolean or not.
  # May speed up computations for models with complex-valued parameters.

 
    
     
  gs = nk.VMC(
      hamiltonian=ha,
      optimizer=op,
      preconditioner=sr,
      variational_state=vs)

  start = time.time()
  gs.run(n_steps, out='RBM')
  end = time.time()

  # import the data from log file
  data=json.load(open("RBM.log"))

  # Extract the relevant information
  # iters_RBM = data["Energy"]["iters"]
  # energy_RBM = data["Energy"]["Mean"]
  # plt.plot(energy_RBM["real"])
  # print("<V|V> =",np.conj(vs.to_array())@vs.to_array())
  # print("<H> =",vs.expect(ha).mean.real)
  # print("Variance =",vs.expect(ha@ha).mean.real-vs.expect(ha).mean.real**2)
  energy = vs.expect(ha).mean.real
  variance = vs.expect(ha@ha).mean.real-vs.expect(ha).mean.real**2
  print([energy, variance])
  return energy, variance;
