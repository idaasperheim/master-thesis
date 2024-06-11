# Importing classes and functions
from classes.molecular_data import MOLECULAR_DATA, STORE_RESULTS
from classes.ansatz import ANSATZ
from algorithms.adapt_vqe import adapt_vqe

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

    # Perform ADAPT-VQE
    h6_adapt_i = adapt_vqe(h6_i.adapt_sparse_operators, h6_i.adapt_qpool, h6_i.sparse_hamiltonian, 
                               h6_i.reference_ket, [1,1,1,1,1,1,0,0,0,0,0,0], h6_i.n_qubits, h6_i.n_electrons, h6_i.n_orbitals,
                               adapt_thres=1e-2)

    # Store the results

    ## ADAPT-VQE
    h6_results.exact_energies_per_distance.append(h6_i.fci_energy)
    h6_results.adapt_converged_energy_per_distance.append(h6_adapt_i.sparse_energies[-1])
    h6_results.adapt_energies_per_distance.append(h6_adapt_i.sparse_energies)
    h6_results.iterations_per_distance.append(h6_adapt_i.iterations) 
    h6_results.states_per_distance.append(h6_adapt_i.curr_state)
    h6_results.ansatz_operators_per_distance.append(h6_adapt_i.ansatz_pool)
    h6_results.ansatz_circuit_per_distance.append(h6_adapt_i.ansatz)
    h6_results.n_gates_per_distance.append(h6_adapt_i.n_gates)
    h6_results.n_cnots_per_distance.append(h6_adapt_i.n_cnots)
    h6_results.parameters_per_distance.append(h6_adapt_i.parameters)
    h6_results.gradients.append(h6_adapt_i.gradients)
    h6_results.max_gradients.append(h6_adapt_i.max_gradients)

h6_adapt_file = open('h6_adapt_filename.obj', 'wb')
pickle.dump(h6_results, h6_adapt_file)