# Importing classes and functions
from classes.molecular_data import MOLECULAR_DATA, STORE_RESULTS
from classes.ansatz import ANSATZ
from algorithms.uccsd_vqe import uccsd_vqe

# Importing libraries
import scipy
import numpy as np
import pickle
import os
import sys
import copy as cp
import pyscf
import openfermion
import openfermionpyscf 
import pennylane as qml
from pennylane import qchem
import qiskit

# Defining distances:
bond_length_interval = 0.1
n_points = 25
bond_lengths = [bond_length_interval * i for i in range(6, n_points+1)]

# Running this model with an array job to reduce computational time on cluster
task_id = int(sys.argv[1])

# Defining storage of molecular result data
h6_results = STORE_RESULTS()

dist = bond_lengths[task_id]
h6_results.distances = dist

print(f'Running for distance: {dist}')

# Get the molecule information
h6_i = MOLECULAR_DATA(['H', 'H', 'H', 'H', 'H', 'H'], dist, [0,6], 0, 0) 

    
# Perform UCCSD-VQE

h6_vqe_i = uccsd_vqe(h6_i.adapt_sparse_operators, h6_i.adapt_qpool, h6_i.sparse_hamiltonian,
                    h6_i.reference_ket, [1,1,1,1,1,1,0,0,0,0,0,0], h6_i.n_qubits, h6_i.n_electrons, h6_i.n_orbitals)

## UCCSD-VQE
h6_results.vqe_energies_per_distance.append(h6_vqe_i.sparse_energies)
h6_results.vqe_iteration_energy_per_distance.append(h6_vqe_i.energy_per_iteration) #energy per optimizer evaluation
h6_results.vqe_iterations_per_distance.append(h6_vqe_i.iter) #optimizer evaluations
h6_results.vqe_parameters_per_distance.append(h6_vqe_i.parameters)
h6_results.vqe_n_cnots.append(h6_vqe_i.n_cnots)
h6_results.vqe_n_gates.append(h6_vqe_i.n_gates)
h6_results.vqe_ansatz_circuit_per_distance.append(h6_vqe_i.ansatz)


h6_uccsd_file = open(f'h6_uccsd_filename_{dist}.obj', 'wb')
pickle.dump(h6_results, h6_uccsd_file)