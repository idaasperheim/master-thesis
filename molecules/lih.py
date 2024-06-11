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
bond_lengths = [bond_length_interval * i for i in range(6, n_points+1)] # Note: shorter set of bond lengths

# Defining storage of molecular result data
lih_results = STORE_RESULTS()
lih_results.distances = bond_lengths

# Performing numerical simulations for LiH molecule
for dist in bond_lengths:
    print(f'Running for distance: {dist}')

    # Get the molecule information
    lih_i = MOLECULAR_DATA(['Li','H'], dist, [1,6], 1, 0) # Note: freezed core

    # Define symmetry-preserving operator pool
    lih_i.reduced_symmetry_pool([1, 2, 5, 6, 9, 10, 12, 13, 16, 17, 18, 19, 21, 22])

    # Perform ADAPT-VQE
    lih_adapt_i = adapt_vqe(lih_i.adapt_sparse_operators, lih_i.adapt_qpool, lih_i.sparse_hamiltonian, 
                            lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals,
                            max_iter = 30)
    
    # Perform symmetry-preserving ADAPT-VQE
    lih_symmetry_adapt_i = adapt_vqe(lih_i.symmetry_spool, lih_i.symmetry_qpool, lih_i.sparse_hamiltonian, 
                            lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals,
                            max_iter = 30)
    
    # Perform QO-ADAPT-VQE
    lih_adapt_qo_i = adapt_vqe(lih_i.adapt_qo_sparse_operators, lih_i.adapt_qo_qpool, lih_i.sparse_hamiltonian, 
                            lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals,
                            max_iter = 30)
    
    # Perform symmetry-preserving QO-ADAPT-VQE
    lih_symmetry_adapt_qo_i = adapt_vqe(lih_i.symmetry_qo_spool, lih_i.symmetry_qo_qpool, lih_i.sparse_hamiltonian, 
                            lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals,
                            max_iter = 30)
    
    # Perform UCCSD-VQE
    lih_vqe_i = uccsd_vqe(lih_i.adapt_sparse_operators, lih_i.adapt_qpool, lih_i.sparse_hamiltonian,
                        lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals)
    
    # Perform QO-VQE
    lih_qovqe_i = uccsd_vqe(lih_i.adapt_qo_sparse_operators, lih_i.adapt_qo_qpool, lih_i.sparse_hamiltonian,
                        lih_i.reference_ket, [1,1,0,0,0,0,0,0,0,0], lih_i.n_qubits, lih_i.n_electrons, lih_i.n_orbitals)
    
    # Store the results
    
    ## FCI

    lih_results.exact_energies_per_distance.append(lih_i.fci_energy)

    ## ADAPT-VQE

    lih_results.adapt_converged_energy_per_distance.append(lih_adapt_i.sparse_energies[-1])
    lih_results.adapt_energies_per_distance.append(lih_adapt_i.sparse_energies)
    lih_results.adapt_symmetry_energies_per_distance.append(lih_symmetry_adapt_i.sparse_energies)
    lih_results.iterations_per_distance.append(lih_adapt_i.iterations)
    lih_results.states_per_distance.append(lih_adapt_i.curr_state)
    lih_results.ansatz_operators_per_distance.append(lih_adapt_i.ansatz_pool)
    lih_results.ansatz_circuit_per_distance.append(lih_adapt_i.ansatz)
    lih_results.n_gates_per_distance.append(lih_adapt_i.n_gates)
    lih_results.n_cnots_per_distance.append(lih_adapt_i.n_cnots)
    lih_results.parameters_per_distance.append(lih_adapt_i.parameters)
    lih_results.gradients.append(lih_adapt_i.gradients)
    lih_results.max_gradients.append(lih_adapt_i.max_gradients)

    ## QO-ADAPT-VQE

    lih_results.adapt_qo_converged_energy_per_distance.append(lih_adapt_qo_i.sparse_energies[-1]) 
    lih_results.adapt_qo_energies_per_distance.append(lih_adapt_qo_i.sparse_energies) 
    lih_results.adapt_qo_symmetry_energies_per_distance.append(lih_symmetry_adapt_qo_i.sparse_energies)
    lih_results.iterations_qo_per_distance.append(lih_adapt_qo_i.iterations)
    lih_results.states_qo_per_distance.append(lih_adapt_qo_i.curr_state)
    lih_results.ansatz_operators_qo_per_distance.append(lih_adapt_qo_i.ansatz_pool)
    lih_results.ansatz_circuit_qo_per_distance.append(lih_adapt_qo_i.ansatz)
    lih_results.n_gates_qo_per_distance.append(lih_adapt_qo_i.n_gates)
    lih_results.n_cnots_qo_per_distance.append(lih_adapt_qo_i.n_cnots)
    lih_results.parameters_qo_per_distance.append(lih_adapt_qo_i.parameters)
    lih_results.gradients_qo.append(lih_adapt_qo_i.gradients)
    lih_results.max_gradients_qo.append(lih_adapt_qo_i.max_gradients)

    ## VQE

    lih_results.vqe_energies_per_distance.append(lih_vqe_i.sparse_energies)
    lih_results.vqe_iteration_energy_per_distance.append(lih_vqe_i.energy_per_iteration) #energy per optimizer evaluation
    lih_results.vqe_iterations_per_distance.append(lih_vqe_i.iter) #optimizer evaluations
    lih_results.vqe_parameters_per_distance.append(lih_vqe_i.parameters)
    lih_results.vqe_n_cnots.append(lih_vqe_i.n_cnots)
    lih_results.vqe_n_gates.append(lih_vqe_i.n_gates)

    ## QO-VQE

    lih_results.qovqe_energies_per_distance.append(lih_qovqe_i.sparse_energies)
    lih_results.qovqe_iteration_energy_per_distance.append(lih_qovqe_i.energy_per_iteration) #energy per optimizer evaluation
    lih_results.qovqe_iterations_per_distance.append(lih_qovqe_i.iter) #optimizer evaluations
    lih_results.qovqe_parameters_per_distance.append(lih_qovqe_i.parameters)
    lih_results.qovqe_n_cnots.append(lih_qovqe_i.n_cnots)
    lih_results.qovqe_n_gates.append(lih_qovqe_i.n_gates)

lih_file = open('lih_filename.obj', 'wb')
pickle.dump(lih_results, lih_file)