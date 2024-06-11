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
n_points = 35
bond_lengths = [bond_length_interval * i for i in range(3, n_points+1)]


# Defining storage of molecular result data
h4_results = STORE_RESULTS()
h4_results.distances = bond_lengths

# Performing ADAPT-VQE for H4 molecule
for dist in bond_lengths:
    print(f'Running for distance: {dist}')

    # Get the molecule information
    h4_i = MOLECULAR_DATA(['H', 'H', 'H', 'H'], dist, [0,4], 0, 0) 

    # Perform ADAPT-VQE
    h4_adapt_i = adapt_vqe(h4_i.adapt_sparse_operators, h4_i.adapt_qpool, h4_i.sparse_hamiltonian, 
                               h4_i.reference_ket, [1,1,1,1,0,0,0,0], h4_i.n_qubits, h4_i.n_electqons, h4_i.n_orbitals)
    
    # Perform QO-ADAPT-VQE
    h4_adapt_qo_i = adapt_vqe(h4_i.adapt_qo_sparse_operators, h4_i.adapt_qo_qpool, h4_i.sparse_hamiltonian, 
                                h4_i.reference_ket, [1,1,1,1,0,0,0,0], h4_i.n_qubits, h4_i.n_electrons, h4_i.n_orbitals)
    
    # Perform UCCSD-VQE
    h4_vqe_i = uccsd_vqe(h4_i.adapt_sparse_operators, h4_i.adapt_qpool, h4_i.sparse_hamiltonian,
                        h4_i.reference_ket, [1,1,1,1,0,0,0,0], h4_i.n_qubits, h4_i.n_electrons, h4_i.n_orbitals)
    
    # Perform QO-UCCSD-VQE
    h4_qovqe_i = uccsd_vqe(h4_i.adapt_qo_sparse_operators, h4_i.adapt_qo_qpool, h4_i.sparse_hamiltonian,
                        h4_i.reference_ket, [1,1,1,1,0,0,0,0], h4_i.n_qubits, h4_i.n_electrons, h4_i.n_orbitals)

    # Store the results

    ## ADAPT-VQE and QO-ADAPT-VQE
    h4_results.exact_energies_per_distance.append(h4_i.fci_energy)
    h4_results.adapt_converged_energy_per_distance.append(h4_adapt_i.sparse_energies[-1])
    h4_results.adapt_energies_per_distance.append(h4_adapt_i.sparse_energies)
    h4_results.adapt_qo_energies_per_distance.append(h4_adapt_qo_i.sparse_energies)
    h4_results.adapt_qo_converged_energy_per_distance.append(h4_adapt_qo_i.sparse_energies[-1])
    h4_results.iterations_per_distance.append(h4_adapt_i.iterations) 
    h4_results.iterations_qo_per_distance.append(h4_adapt_qo_i.iterations)
    h4_results.states_per_distance.append(h4_adapt_i.curr_state)
    h4_results.states_qo_per_distance.append(h4_adapt_qo_i.curr_state)
    h4_results.ansatz_operators_per_distance.append(h4_adapt_i.ansatz_pool)
    h4_results.ansatz_circuit_per_distance.append(h4_adapt_i.ansatz)
    h4_results.ansatz_operators_qo_per_distance.append(h4_adapt_qo_i.ansatz_pool)
    h4_results.ansatz_circuit_qo_per_distance.append(h4_adapt_qo_i.ansatz)
    h4_results.n_gates_per_distance.append(h4_adapt_i.n_gates)
    h4_results.n_gates_qo_per_distance.append(h4_adapt_qo_i.n_gates)
    h4_results.n_cnots_per_distance.append(h4_adapt_i.n_cnots)
    h4_results.n_cnots_qo_per_distance.append(h4_adapt_qo_i.n_cnots)
    h4_results.parameters_per_distance.append(h4_adapt_i.parameters)
    h4_results.parameters_qo_per_distance.append(h4_adapt_qo_i.parameters)

    h4_results.gradients.append(h4_adapt_i.gradients)
    h4_results.gradients_qo.append(h4_adapt_qo_i.gradients)
    h4_results.max_gradients.append(h4_adapt_i.max_gradients)
    h4_results.max_gradients_qo.append(h4_adapt_qo_i.max_gradients)

    ## UCCSD-VQE
    h4_results.vqe_energies_per_distance.append(h4_vqe_i.sparse_energies)
    h4_results.vqe_iteration_energy_per_distance.append(h4_vqe_i.energy_per_iteration) #energy per optimizer evaluation
    h4_results.vqe_iterations_per_distance.append(h4_vqe_i.iter) #optimizer evaluations
    
    h4_results.vqe_parameters_per_distance.append(h4_vqe_i.parameters)
    h4_results.vqe_n_cnots.append(h4_vqe_i.n_cnots)
    h4_results.vqe_n_gates.append(h4_vqe_i.n_gates)
    h4_results.vqe_ansatz_circuit_per_distance.append(h4_vqe_i.ansatz)

    ## QO-UCCSD-VQE
    h4_results.qovqe_energies_per_distance.append(h4_qovqe_i.sparse_energies)
    h4_results.qovqe_iteration_energy_per_distance.append(h4_qovqe_i.energy_per_iteration) #energy per optimizer evaluation
    h4_results.qovqe_iterations_per_distance.append(h4_qovqe_i.iter) #optimizer evaluations

    h4_results.qovqe_parameters_per_distance.append(h4_qovqe_i.parameters)
    h4_results.qovqe_n_cnots.append(h4_qovqe_i.n_cnots)
    h4_results.qovqe_n_gates.append(h4_qovqe_i.n_gates)
    h4_results.qovqe_ansatz_circuit_per_distance.append(h4_qovqe_i.ansatz)

h4_file = open('h4_filename.obj', 'wb')
pickle.dump(h4_results, h4_file)