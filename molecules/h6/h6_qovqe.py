# Importing classes and functions
from classes.molecular_data import MOLECULAR_DATA, STORE_RESULTS
from classes.ansatz import ANSATZ
from algorithms.uccsd_vqe import uccsd_vqe

# Importing libraries
import scipy
import numpy as np
import pickle
import os
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


# Defining storage of molecular result data
h6_results = STORE_RESULTS()
h6_results.distances = bond_lengths

# Performing ADAPT-VQE for h6 molecule
for dist in bond_lengths:
    print(f'Running for distance: {dist}')

    # Get the molecule information
    h6_i = MOLECULAR_DATA(['H', 'H', 'H', 'H', 'H', 'H'], dist, [0,6], 0, 0) 
    
    # Perform QO-UCCSD-VQE
    h6_qovqe_i = uccsd_vqe(h6_i.adapt_qo_sparse_operators, h6_i.adapt_qo_qpool, h6_i.sparse_hamiltonian,
                        h6_i.reference_ket, [1,1,1,1,1,1,0,0,0,0,0,0], h6_i.n_qubits, h6_i.n_electrons, h6_i.n_orbitals)

    # Store the results

    ## QO-UCCSD-VQE

    h6_results.qovqe_energies_per_distance.append(h6_qovqe_i.sparse_energies)
    h6_results.qovqe_iteration_energy_per_distance.append(h6_qovqe_i.energy_per_iteration) #energy per optimizer evaluation
    h6_results.qovqe_iterations_per_distance.append(h6_qovqe_i.iter) #optimizer evaluations

    h6_results.qovqe_parameters_per_distance.append(h6_qovqe_i.parameters)
    h6_results.qovqe_n_cnots.append(h6_qovqe_i.n_cnots)
    h6_results.qovqe_n_gates.append(h6_qovqe_i.n_gates)
    h6_results.qovqe_ansatz_circuit_per_distance.append(h6_qovqe_i.ansatz)

h6_qouccsd_file = open('h6_qouccsd_filename.obj', 'wb')
pickle.dump(h6_results, h6_qouccsd_file)