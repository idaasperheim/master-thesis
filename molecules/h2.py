# Importing classes and functions
from classes.molecular_data import MOLECULAR_DATA, STORE_RESULTS
from classes.ansatz import ANSATZ
from algorithms.adapt_vqe import adapt_vqe
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
bond_lengths = [bond_length_interval * i for i in range(3, n_points+1)]

# Defining storage of molecular result data
h2_results = STORE_RESULTS()
h2_results.distances = bond_lengths

# Performing ADAPT-VQE for h2 molecule
for dist in bond_lengths:
    print(f'Running for distance: {dist}')

    # Get the molecule information
    h2_i = MOLECULAR_DATA(['H', 'H'], dist, [0,2], 0, 0) # Do not freeze core

    # Perform ADAPT-VQE
    h2_adapt_i = adapt_vqe(h2_i.adapt_sparse_operators, h2_i.adapt_qpool, h2_i.sparse_hamiltonian, 
                               h2_i.reference_ket, [1,1,0,0], h2_i.n_qubits, h2_i.n_electrons, h2_i.n_orbitals)
    
    # Perform QO-ADAPT-VQE
    h2_adapt_qo_i = adapt_vqe(h2_i.adapt_qo_sparse_operators, h2_i.adapt_qo_qpool, h2_i.sparse_hamiltonian, 
                                h2_i.reference_ket, [1,1,0,0], h2_i.n_qubits, h2_i.n_electrons, h2_i.n_orbitals)
    
    # Perform UCCSD-VQE
    h2_vqe_i = uccsd_vqe(h2_i.adapt_sparse_operators, h2_i.adapt_qpool, h2_i.sparse_hamiltonian,
                        h2_i.reference_ket, [1,1,0,0], h2_i.n_qubits, h2_i.n_electrons, h2_i.n_orbitals)
    
    # Perform QO-UCCSD-VQE
    h2_qovqe_i = uccsd_vqe(h2_i.adapt_qo_sparse_operators, h2_i.adapt_qo_qpool, h2_i.sparse_hamiltonian,
                        h2_i.reference_ket, [1,1,0,0], h2_i.n_qubits, h2_i.n_electrons, h2_i.n_orbitals)

    # Store the results
    
    ## FCI
    h2_results.exact_energies_per_distance.append(h2_i.fci_energy)
    
    ## ADAPT-VQE 
    h2_results.adapt_converged_energy_per_distance.append(h2_adapt_i.sparse_energies[-1])
    h2_results.adapt_energies_per_distance.append(h2_adapt_i.sparse_energies)
    h2_results.iterations_per_distance.append(h2_adapt_i.iterations) 
    h2_results.states_per_distance.append(h2_adapt_i.curr_state)
    h2_results.ansatz_operators_per_distance.append(h2_adapt_i.ansatz_pool)
    h2_results.ansatz_circuit_per_distance.append(h2_adapt_i.ansatz)
    h2_results.n_gates_per_distance.append(h2_adapt_i.n_gates)
    h2_results.n_cnots_per_distance.append(h2_adapt_i.n_cnots)
    h2_results.parameters_per_distance.append(h2_adapt_i.parameters)
    h2_results.gradients.append(h2_adapt_i.gradients)
    h2_results.max_gradients.append(h2_adapt_i.max_gradients)

    ## QO-ADAPT-VQE

    h2_results.adapt_qo_energies_per_distance.append(h2_adapt_qo_i.sparse_energies)
    h2_results.adapt_qo_converged_energy_per_distance.append(h2_adapt_qo_i.sparse_energies[-1])
    h2_results.iterations_qo_per_distance.append(h2_adapt_qo_i.iterations)
    h2_results.states_qo_per_distance.append(h2_adapt_qo_i.curr_state)
    h2_results.ansatz_operators_qo_per_distance.append(h2_adapt_qo_i.ansatz_pool)
    h2_results.ansatz_circuit_qo_per_distance.append(h2_adapt_qo_i.ansatz)
    h2_results.n_gates_qo_per_distance.append(h2_adapt_qo_i.n_gates)
    h2_results.n_cnots_qo_per_distance.append(h2_adapt_qo_i.n_cnots)
    h2_results.parameters_qo_per_distance.append(h2_adapt_qo_i.parameters)
    h2_results.gradients_qo.append(h2_adapt_qo_i.gradients)
    h2_results.max_gradients_qo.append(h2_adapt_qo_i.max_gradients)

    ## UCCSD-VQE
    h2_results.vqe_energies_per_distance.append(h2_vqe_i.sparse_energies)
    h2_results.vqe_iteration_energy_per_distance.append(h2_vqe_i.energy_per_iteration) #energy per optimizer evaluation
    h2_results.vqe_iterations_per_distance.append(h2_vqe_i.iter) #optimizer evaluations
    h2_results.vqe_parameters_per_distance.append(h2_vqe_i.parameters)
    h2_results.vqe_n_cnots.append(h2_vqe_i.n_cnots)
    h2_results.vqe_n_gates.append(h2_vqe_i.n_gates)
    h2_results.vqe_ansatz_circuit_per_distance.append(h2_vqe_i.ansatz)

    ## QO-UCCSD-VQE
    h2_results.qovqe_energies_per_distance.append(h2_qovqe_i.sparse_energies)
    h2_results.qovqe_iteration_energy_per_distance.append(h2_qovqe_i.energy_per_iteration) #energy per optimizer evaluation
    h2_results.qovqe_iterations_per_distance.append(h2_qovqe_i.iter) #optimizer evaluations
    h2_results.qovqe_parameters_per_distance.append(h2_qovqe_i.parameters)
    h2_results.qovqe_n_cnots.append(h2_qovqe_i.n_cnots)
    h2_results.qovqe_n_gates.append(h2_qovqe_i.n_gates)
    h2_results.qovqe_ansatz_circuit_per_distance.append(h2_qovqe_i.ansatz)

h2_file = open('h2_filename.obj', 'wb')
pickle.dump(h2_results, h2_file)