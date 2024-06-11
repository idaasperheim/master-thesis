# Importing classes
from classes.molecular_data import MOLECULAR_DATA
from classes.ansatz import ANSATZ

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

def uccsd_vqe(operators, pool, sparse_hamiltonian, reference_ket, occ_basis, n_qubits, n_electrons, n_orbitals):
    '''
    Function to run the UCCSD-VQE algorithm for a molecule of interest.
    '''
    
    curr_state = 1.0*reference_ket
    parameters = []
    curr_operators = []

    vqe_model = ANSATZ(sparse_hamiltonian, curr_operators, reference_ket, 
                                  parameters, n_qubits, n_electrons, n_orbitals, occ_basis)
    
    h_ref_state = sparse_hamiltonian.dot(vqe_model.curr_state)
    e_ref = vqe_model.curr_state.T.conj().dot(h_ref_state)[0,0].real
    vqe_model.sparse_energies.append(e_ref)
    
    for op in operators:

        vqe_model.ansatz_ops.append(op)
        vqe_model.parameters.append(0)
    
        optimize_result = scipy.optimize.minimize(vqe_model.energy, parameters, method='BFGS', callback=vqe_model.callback)
        parameters = list(optimize_result['x'])

        curr_state = vqe_model.prepare_state(parameters)
        vqe_model.curr_state = curr_state
        vqe_model.parameters = parameters
        
        vqe_model.sparse_energies.append(vqe_model.curr_energy)

    vqe_model.n_params = len(vqe_model.parameters)
    vqe_model.ansatz_pool = pool
    vqe_model.create_final_circuit()
    
    return vqe_model