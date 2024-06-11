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


class ANSATZ():
    def __init__(self, _H, _G, _ref, _params, _n_qubits, _n_electrons, _n_spatial_orbitals, _occ_basis):
        """
        _H      : sparse matrix
        _G_ops  : list of sparse matrices - each corresponds to a variational parameter
        _ref    : reference state vector
        _params : initialized list of parameters
        """

        self.H = _H
        self.ansatz_ops = _G
        self.ref = cp.deepcopy(_ref)
        self.parameters = _params
        self.n_params = len(self.parameters)
        self.hilb_dim = self.H.shape[0] 
        
        self.iter = 0 #use for uccsd-vqe
        self.energy_per_iteration = [] #use for uccsd-vqe
        self.psi_norm = 1.0

        #For storing circuit ansatz
        self.n_qubits = _n_qubits
        self.n_electrons = _n_electrons
        self.n_spatial_orbitals = _n_spatial_orbitals
        
        self.n_gates = [0]
        self.n_cnots = [0]
        self.make_hartree_state(_occ_basis)

        #For storing results
        self.curr_state = _ref
        self.iterations = []
        self.sparse_energies = []
        self.ansatz_pool = []

        self.gradients = []
        self.max_gradients=[]
    
    def callback(self, x):
        """
        Callback function for the optimizer
        """
        self.iter += 1
        self.energy_per_iteration.append(self.curr_energy)
    
    def make_hartree_state(self, occ_basis):
        """
        Create the Hartree-Fock circuit state
        """
        circuit = qiskit.QuantumCircuit(self.n_qubits)
        for count, value in enumerate(occ_basis): 
            if value==1:
                circuit.x(count)
        self.ansatz = circuit
    

    def energy(self,parameters):
        """
        Calculate the energy of the current state
        """
        new_state = self.prepare_state(parameters)
        assert(new_state.transpose().conj().dot(new_state).toarray()[0][0]-1<0.0000001)
        energy = new_state.transpose().conj().dot(self.H.dot(new_state))[0,0]
        assert(np.isclose(energy.imag,0))
        self.curr_energy = energy.real
        return energy.real

    def prepare_state(self,parameters):
        """ 
        Prepare state: exp{A1}exp{A2}exp{A3}...exp{An}|ref>
        """
        new_state = self.ref * 1.0
        for k in range(0, len(parameters)):
            new_state = scipy.sparse.linalg.expm_multiply((parameters[k]*self.ansatz_ops[k]), new_state)
        return new_state
    
    

    def pauli_gate(self, operator, parameter):
        '''
        Creates the circuit for applying e^ (j * operator * parameter), for 'operator'
        a single Pauli string.
        Uses little endian endian, as Qiskit requires.

        Arguments:
        operator (union[openfermion.QubitOperator, openfermion.FermionOperator,
        openfermion.InteractionOperator]): the operator to be simulated
        parameter (qiskit.circuit.parameter): the variational parameter
        circuit (qiskit.circuit.QuantumCircuit): the circuit to add the gates to
        no_qubits (int): the number of qubits in the circuit
        '''
        no_gates = 0
        no_cnots = 0

        circuit = self.ansatz
        no_qubits = self.n_qubits

        # Iterate over all Pauli strings in the operator
        for no in range(len(operator.terms)):

            # Transform InteractionOperator into FermionOperator
            if isinstance(operator, openfermion.InteractionOperator):
                operator = openfermion.get_fermion_operator(operator)

            # Transform FermionOperator into QubitOperator using the Jordan Wigner transformation
            if isinstance(operator,openfermion.FermionOperator):
                operator = openfermion.jordan_wigner(operator)

            # 1. Isolate one Pauli string 
            # If we have a sum 0 will only take the first, while 1 the second. no is the index of the Pauli string,
            pauli_string = list(operator.terms.keys())[no] 

            # Store the qubit indices involved in this particular Pauli string for computing parity
            involved_qubits = []

            # 2. Perform basis rotations (if necessary)
            for pauli in pauli_string:

                # Get the index of the qubit the Pauli operator within the Pauli string acts on
                index = pauli[0]
                involved_qubits.append(index)

                # Get the Pauli operator (X, Y or Z)
                pauli_operator = pauli[1]

                # Rotate to basis if X or Y
                if pauli_operator == "X":
                    circuit.h(no_qubits - 1 - index)
                    no_gates += 1

                if pauli_operator == "Y":
                    circuit.rx(np.pi/2,no_qubits - 1 - index)
                    no_gates += 1

            # 3. Compute parity of the qubits involved in the Pauli string
            for i in range(len(involved_qubits)-1):

                control = involved_qubits[i]
                target = involved_qubits[i+1]

                circuit.cx(no_qubits - 1 - control,no_qubits - 1 - target)
                no_gates += 1
                no_cnots += 1
  
            # 4. Apply e^(-i*Z*parameter) = Rz(-parameter*2) rotation qubit
            rot_qubit = max(involved_qubits)
            circuit.rz(-2 * parameter,no_qubits - 1 - rot_qubit)
            no_gates += 1

            # 5. Uncompute parity of the qubits involved in the Pauli string
            for i in range(len(involved_qubits)-2,-1,-1):

                control = involved_qubits[i]
                target = involved_qubits[i+1]

                circuit.cx(no_qubits - 1 - control,no_qubits - 1 - target)
                no_gates += 1
                no_cnots += 1

            # 6. Undo basis rotations
            for pauli in pauli_string:

                # Get the index of the qubit the Pauli operator within the Pauli string acts on
                index = pauli[0]

                # Get the Pauli operator (X,Y or Z)
                pauli_operator = pauli[1]

                # Rotate to basis if X or Y
                if pauli_operator == "X":
                    circuit.h(no_qubits - 1 - index)
                    no_gates += 1

                if pauli_operator == "Y":
                    circuit.rx(-np.pi/2,no_qubits - 1 - index)
                    no_gates += 1

        stored_n_gates = sum(self.n_gates)
        stored_n_cnots = sum(self.n_cnots)
        self.n_gates.append(no_gates + stored_n_gates)
        self.n_cnots.append(no_cnots + stored_n_cnots)
        self.ansatz = circuit
    
    def create_final_circuit(self):
        for i in range(self.n_params):
            self.pauli_gate(self.ansatz_pool[i], self.parameters[i])