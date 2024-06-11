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


#TODO: implement calculate_gradient inside the ANSATZ class

def calculate_gradient(sparse_operator, curr_state, h_curr_state):
    '''
    Function to calculate the gradient of a given current state, curr_state, with respect to a sparse operator, sparse_operator.
    '''
    gi = 2*(h_curr_state.transpose().conj().dot(sparse_operator.dot(curr_state))) 
    assert(gi.shape == (1,1))
    gi = gi[0,0]
    assert(np.isclose(gi.imag,0)) 
    gi = gi.real

    return gi

def adapt_vqe(operators, pool, sparse_hamiltonian, reference_ket, occ_basis, n_qubits, n_electrons, n_orbitals,
                  max_iter = 10, adapt_thres = 1e-3):
    """
    Function to run the ADAPT-VQE algorithm of a molecule of interest.
    """

    parameters = []
    sparse_energies = []

    ansatz_ops = []
    ansatz_pool = []
    ansatz_indicies = []
    
    curr_state = 1.0*reference_ket

    # Initialize ansatz
    trial_model = ANSATZ(sparse_hamiltonian, ansatz_ops, reference_ket, 
                                parameters, n_qubits, n_electrons, n_orbitals, occ_basis)
    
    for n_iter in range(0, max_iter):
        print(" Iteration: %4i" % n_iter)

        next_index = None
        next_deriv = 0
        curr_norm = 0

        h_curr_state = sparse_hamiltonian.dot(curr_state)
        e_curr = curr_state.T.conj().dot(h_curr_state)[0,0]

        if n_iter == 0:
            print(f'Reference energy: {e_curr.real}')
            sparse_energies.append(e_curr.real)
       
        var = h_curr_state.T.conj().dot(h_curr_state)[0,0] - e_curr**2
        uncertainty = np.sqrt(var.real)

        print(" Variance:    %12.8f" % var.real)
        print(" Uncertainty: %12.8f" % uncertainty)

        # Calculate gradient for all operators in operator pool
        intermediate_g = []
        for oi in range(len(operators)):
            gi = calculate_gradient(operators[oi], curr_state, h_curr_state)
            intermediate_g.append(gi)
            print(f"Gradient to operator {oi}: {gi}")
            curr_norm += gi*gi

            if abs(gi) > abs(next_deriv):
                next_deriv = gi
                next_index = oi
        
        
        curr_norm = np.sqrt(curr_norm)
        print("Current norm: ", curr_norm)
    
        # Find the index of the maximum gradient in the pool, which gives the operator to add
        max_gi = next_deriv

        # Check convergence
        converged = False
        if curr_norm < adapt_thres:
            converged = True

        if converged:
            print("ADAPT-VQE model converged!")
            break
        
        # Save gradients
        trial_model.max_gradients.append(dict(zip([next_index], [max_gi])))
        trial_model.gradients.append(intermediate_g)


        # Add operator to ansatz
        parameters.append(0)
        ansatz_ops.append(operators[next_index])
        ansatz_pool.append(pool[next_index])
        ansatz_indicies.append(next_index)
        iterations = [iteration for iteration in range(0,n_iter+2)]

        # Update class parameters
        trial_model.iterations = iterations
        trial_model.ansatz_pool = ansatz_pool
        trial_model.ansatz_ops = ansatz_ops
        trial_model.parameters = parameters

        # Optimize parameters
        optimize_result = scipy.optimize.minimize(trial_model.energy, parameters, method='BFGS') 
        parameters = list(optimize_result['x'])
        print(parameters)

        # Update parameters
        trial_model.parameters = parameters
        trial_model.n_params = len(trial_model.parameters)
       
        # Update energy and state after optimization
        curr_state = trial_model.prepare_state(parameters)
        trial_model.curr_state = curr_state

        sparse_energies.append(trial_model.curr_energy)
        trial_model.sparse_energies = sparse_energies


    print("ADAPT-VQE model finished!")
    # Create final quantum circuit
    trial_model.create_final_circuit()
    

    return trial_model 