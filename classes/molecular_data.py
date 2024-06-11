# Import the necessary libraries
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


class MOLECULAR_DATA():
    def __init__(self, _molecule_info, _distance, _active_space, _frozen_orbitals, _virtual_orbitals):
        '''
        Class to get the molecular information for a molecule of interest.
        
        molecule_info: List. [atom_1, atom_2]
        distance: Float. The bond length of the molecule.
        active_space: List. [start, stop]. The active space of the molecule.
        frozen_orbitals: Integer. Number of frozen orbitals core orbitals.
        virutal_orbitals: Integer. Number of frozen virtual orbitals.
        array_geometry: Array. The geometry of the molecule. [x1, y1, z1, x2, y2, z2]

        n_orbitals: Integer. The number of molecular orbitals.
        n_electrons: Integer. The number of electrons in the molecule.
        n_qubits: Integer. The number of qubits in the molecule.

        sparse_hamiltonian: SparseOperator. The sparse Hamiltonian of the molecule.
        qubit_hamiltonian: QubitOperator. The qubit Hamiltonian of the molecule.
        fci_energy: Float. The exact energy of the molecule.
        fci_state: Array. The exact state of the molecule.

        valid_single_excitations: List. The valid single excitations of the molecule.
        valid_double_excitations: List. The valid double excitations of the molecule.

        adapt_qpool: List. The qubit operators for the single and double excitations.
        adapt_qo_qpool: List. The qubit operators for the single and double excitations with the reduced operators.

        adapt_sparse_operators: List. The sparse operators for the single and double excitations.
        adapt_qo_sparse_operators: List. The sparse operators for the single and double excitations with the reduced operators.
        '''
        self.molecule_info = _molecule_info 
        self.distance = _distance
        self.active_space = _active_space
        self.frozen_orbitals = _frozen_orbitals
        self.virtual_orbitals = _virtual_orbitals

        if len(self.molecule_info)==2:
            self.array_geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.distance])
            self.driver_geometry = f'{self.molecule_info[0]} 0.0 0.0 0.0; {self.molecule_info[1]} 0.0 0.0 {self.distance}'
        
        if len(self.molecule_info)==3:
            self.array_geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.distance, 
                                            0.0, 0.0, 2*self.distance])
            self.driver_geometry = f'{self.molecule_info[0]} 0.0 0.0 0.0; {self.molecule_info[1]} 0.0 0.0 {self.distance}; {self.molecule_info[2]} 0.0 0.0 {2*self.distance}'

        if len(self.molecule_info)==4:
            self.array_geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.distance, 
                                            0.0, 0.0, 2*self.distance, 0.0, 0.0, 3*self.distance])
            self.driver_geometry = f'{self.molecule_info[0]} 0.0 0.0 0.0; {self.molecule_info[1]} 0.0 0.0 {self.distance}; {self.molecule_info[2]} 0.0 0.0 {2*self.distance}; {self.molecule_info[3]} 0.0 0.0 {3*self.distance}'

        if len(self.molecule_info)==6:
            self.array_geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.distance, 
                                            0.0, 0.0, 2*self.distance, 0.0, 0.0, 3*self.distance,
                                            0.0, 0.0, 4*self.distance, 0.0, 0.0, 5*self.distance])
            self.driver_geometry = f'{self.molecule_info[0]} 0.0 0.0 0.0; {self.molecule_info[1]} 0.0 0.0 {self.distance}; {self.molecule_info[2]} 0.0 0.0 {2*self.distance}; {self.molecule_info[3]} 0.0 0.0 {3*self.distance}; {self.molecule_info[4]} 0.0 0.0 {4*self.distance}; {self.molecule_info[5]} 0.0 0.0 {5*self.distance}'

        #Initialize the molecule information
        self.get_molecule_info()
        self.get_excitation_index()
        self.create_qubit_excitations()
        self.create_qubit_excitations_qo()
        self.create_sparse_operators()

        self.reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(self.n_electrons)), self.n_qubits)).transpose()


    def get_molecule_info(self):
        '''
        Function to get the molecule information for the molecule of interest.
        Returns the sparse Hamiltonian, qubit Hamiltonian, exact energy and the number of qubits.

        molecule_info: List. [atom_1, atom_2]
        distance: Float. The bond length of the molecule.
        active_space: List. [start, stop]. The active space of the molecule.
        frozen_orbitals: Integer. Number of frozen orbitals.
        virutal_orbitals: Integer. Number of frozen virtual orbitals.
        '''
        # Define molecular geometry.

        if len(self.molecule_info) == 2:
            geometry = [(f'{self.molecule_info[0]}', (0., 0., 0.)), (f'{self.molecule_info[1]}', (0., 0., self.distance))]
        
        if len(self.molecule_info) == 3:
            geometry = [(f'{self.molecule_info[0]}', (0., 0., 0.)), (f'{self.molecule_info[1]}', (0., 0., self.distance)), 
                        (f'{self.molecule_info[2]}', (0., 0., 2*self.distance))]


        if len(self.molecule_info) == 4:
            geometry = [(f'{self.molecule_info[0]}', (0., 0., 0.)), (f'{self.molecule_info[1]}', (0., 0., self.distance)), 
                        (f'{self.molecule_info[2]}', (0., 0., 2*self.distance)), (f'{self.molecule_info[3]}', (0., 0., 3*self.distance))]


        if len(self.molecule_info) == 6:
            geometry = [(f'{self.molecule_info[0]}', (0., 0., 0.)), (f'{self.molecule_info[1]}', (0., 0., self.distance)), 
                        (f'{self.molecule_info[2]}', (0., 0., 2*self.distance)), (f'{self.molecule_info[3]}', (0., 0., 3*self.distance)),
                        (f'{self.molecule_info[4]}', (0., 0., 4*self.distance)), (f'{self.molecule_info[5]}', (0., 0., 5*self.distance))]


        basis = 'sto-3g'
        multiplicity = 1
        description = str(round(self.distance, 2))
        active_space_start = self.active_space[0]
        active_space_stop = self.active_space[1]

        # Generate and populate instance of MolecularData.

        package="pyscf"
        outpath="."
        name="molecule"
        filename = name + "_" + package.lower() + "_" + basis.strip()
        path_to_file = os.path.join(outpath.strip(), filename)
        molecule = openfermion.MolecularData(geometry, basis, multiplicity, description=description, filename=path_to_file)
        molecule = openfermionpyscf.run_pyscf(molecule, run_fci=1, verbose=0)
        self.mol_fci_energy = molecule.fci_energy
        molecule.load()

        # Generate molecular information
        n_orbitals = molecule.n_orbitals - self.frozen_orbitals - self.virtual_orbitals
        n_electrons = molecule.n_electrons - 2*self.frozen_orbitals
        n_qubits = molecule.n_qubits - 2*self.frozen_orbitals - 2*self.virtual_orbitals

        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.n_qubits = n_qubits

        # Get the Hamiltonian in an active space.
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=range(active_space_start),
        active_indices=range(active_space_start, active_space_stop))

        # Map operator to fermions and qubits.
        fermion_hamiltonian = openfermion.get_fermion_operator(molecular_hamiltonian)
        qubit_hamiltonian = openfermion.jordan_wigner(fermion_hamiltonian)
        qubit_hamiltonian.compress()

        # Get sparse operator and ground state energy.
        sparse_hamiltonian = openfermion.get_sparse_operator(qubit_hamiltonian)
        energy, state = openfermion.get_ground_state(sparse_hamiltonian)

        self.sparse_hamiltonian = sparse_hamiltonian
        self.qubit_hamiltonian = qubit_hamiltonian
        self.fci_energy = energy
        self.fci_state = state


    def get_excitation_index(self):
        '''
        Function to get the excitation index for the molecule of interest.
        Returns the valid single and double excitations.

        molecule_info: List. [atom_1, atom_2]
        array_geometry: Array. The geometry of the molecule. [x1, y1, z1, x2, y2, z2]
        '''

        # compute molecular Hamiltonian in the STO-3G basis
        H, n_qubits = qchem.molecular_hamiltonian(self.molecule_info, 
                                                self.array_geometry,
                                                active_electrons = self.n_electrons,
                                                active_orbitals = self.n_orbitals)

        # obtain electronic excitations, restricted to single and double excitations
        singles, doubles = qchem.excitations(self.n_electrons, self.n_qubits)
        print(f"Total number of excitations = {len(singles) + len(doubles)}")
        self.valid_single_excitations = singles
        self.valid_double_excitations = doubles
    
    def create_qubit_excitations_qo(self):
        '''
        Function to obtain the qubit operators with reduced Pauli weight corresponding to physically allowed qubit excitations.
        '''
        qubit_operators = []
        for single in self.valid_single_excitations:
            qubit_operators.append(1j/2*(openfermion.QubitOperator(f'X{single[1]} Y{single[0]}') - openfermion.QubitOperator(f'Y{single[1]} X{single[0]}')))
    
        for double in self.valid_double_excitations:
            qubit_operators.append(1j/8*(openfermion.QubitOperator(f'Y{double[3]} Y{double[2]} X{double[1]} Y{double[0]}') - openfermion.QubitOperator(f'X{double[3]} X{double[2]} X{double[1]} Y{double[0]}')
                                                   - openfermion.QubitOperator(f'Y{double[3]} X{double[2]} Y{double[1]} Y{double[0]}') + openfermion.QubitOperator(f'Y{double[3]} Y{double[2]} Y{double[1]} X{double[0]}')
                                                   - openfermion.QubitOperator(f'X{double[3]} Y{double[2]} Y{double[1]} Y{double[0]}') - openfermion.QubitOperator(f'X{double[3]} X{double[2]} Y{double[1]} X{double[0]}')
                                                   + openfermion.QubitOperator(f'Y{double[3]} X{double[2]} X{double[1]} X{double[0]}') + openfermion.QubitOperator(f'X{double[3]} Y{double[2]} X{double[1]} X{double[0]}')))
    
        self.adapt_qo_qpool = qubit_operators
    
    def create_qubit_excitations(self):
        '''
        Function to obtain the fermionic operators of physically allowed excitations mapped onto qubits. The encoded operators are obtained with Jordan-Wigner mapping.
        '''
        qubit_operators = []
        for single in self.valid_single_excitations:
            
            P = 1
            for p in range(min(single[1], single[0])+1, max(single[1], single[0])): #between i+1 and k-1
                P *= openfermion.QubitOperator(f'Z{p}')
            
            qubit_operators.append(1j/2*P*(openfermion.QubitOperator(f'X{single[1]} Y{single[0]}') - openfermion.QubitOperator(f'Y{single[1]} X{single[0]}')))

                

        for double in self.valid_double_excitations:
    
            P = 1
            for p in range(min(double[3], double[2])+1, max(double[3], double[2])): #between k+1 and l-1
                P *= openfermion.QubitOperator(f'Z{p}')

            for p in range(min(double[1], double[0])+1, max(double[1], double[0])): #between i+1 and j-1
                P *= openfermion.QubitOperator(f'Z{p}')

            qubit_operators.append(1j/8*P*(openfermion.QubitOperator(f'Y{double[3]} Y{double[2]} X{double[1]} Y{double[0]}') - openfermion.QubitOperator(f'X{double[3]} X{double[2]} X{double[1]} Y{double[0]}')
                                                   - openfermion.QubitOperator(f'Y{double[3]} X{double[2]} Y{double[1]} Y{double[0]}') + openfermion.QubitOperator(f'Y{double[3]} Y{double[2]} Y{double[1]} X{double[0]}')
                                                   - openfermion.QubitOperator(f'X{double[3]} Y{double[2]} Y{double[1]} Y{double[0]}') - openfermion.QubitOperator(f'X{double[3]} X{double[2]} Y{double[1]} X{double[0]}')
                                                   + openfermion.QubitOperator(f'Y{double[3]} X{double[2]} X{double[1]} X{double[0]}') + openfermion.QubitOperator(f'X{double[3]} Y{double[2]} X{double[1]} X{double[0]}')))
    
        self.adapt_qpool = qubit_operators
    

    def create_sparse_operators(self):
        '''
        Function to obtain the sparse representation of the qubit operators. 
        '''
        adapt_sparse_operators = []
        adapt_qo_sparse_operators = []
        for operator in self.adapt_qpool:
            adapt_sparse_operators.append(openfermion.linalg.get_sparse_operator(operator, self.n_qubits))
        
        for operator_qo in self.adapt_qo_qpool:
            adapt_qo_sparse_operators.append(openfermion.linalg.get_sparse_operator(operator_qo, self.n_qubits))

        self.adapt_sparse_operators = adapt_sparse_operators
        self.adapt_qo_sparse_operators = adapt_qo_sparse_operators

    def reduced_symmetry_pool(self, operator_list):
        '''
        Function to obtain the reduced symmetry pool for the molecular system.
        '''
        reduced_qpool = []
        reduced_qo_qpool = []

        reduced_spool = []
        reduced_qo_spool = []

        all_indicies = list(range(len(self.adapt_qpool)))

        for index in all_indicies:
            if index in operator_list:
                continue
            else:
                reduced_qpool.append(self.adapt_qpool[index])
                reduced_qo_qpool.append(self.adapt_qo_qpool[index])
                reduced_spool.append(self.adapt_sparse_operators[index])
                reduced_qo_spool.append(self.adapt_qo_sparse_operators[index])

        self.symmetry_qpool = reduced_qpool
        self.symmetry_qo_qpool = reduced_qo_qpool
        self.symmetry_spool = reduced_spool
        self.symmetry_qo_spool = reduced_qo_spool

class STORE_RESULTS():
    def __init__(self):

        # FCI

        self.exact_energies_per_distance = []
        self.distances = []

        # ADAPT-VQE

        self.adapt_energies_per_distance = []
        self.adapt_symmetry_energies_per_distance = []
        self.adapt_converged_energy_per_distance = []
        self.iterations_per_distance = []
        self.states_per_distance = []
        self.parameters_per_distance = []
        self.ansatz_operators_per_distance = []
        self.ansatz_circuit_per_distance = []
        self.n_gates_per_distance = []
        self.n_cnots_per_distance = []
        self.gradients = []
        self.max_gradients = []

        # QO-ADAPT-VQE

        self.adapt_qo_energies_per_distance = []
        self.adapt_qo_symmetry_energies_per_distance = []
        self.adapt_qo_converged_energy_per_distance = []
        self.iterations_qo_per_distance = []
        self.states_qo_per_distance = []
        self.parameters_qo_per_distance = []
        self.ansatz_operators_qo_per_distance = []
        self.ansatz_circuit_qo_per_distance = []
        self.n_gates_qo_per_distance = []
        self.n_cnots_qo_per_distance = []
        self.gradients_qo = []
        self.max_gradients_qo = []

        # VQE

        self.vqe_energies_per_distance = []
        self.vqe_iteration_energy_per_distance = []
        self.vqe_iterations_per_distance = []
        self.vqe_parameters_per_distance = []
        self.vqe_n_cnots = []
        self.vqe_n_gates = []
        self.vqe_ansatz_circuit_per_distance = []

        # QO-UCCSD-VQE

        self.qovqe_energies_per_distance = []
        self.qovqe_iteration_energy_per_distance = []
        self.qovqe_iterations_per_distance = []
        self.qovqe_parameters_per_distance = []
        self.qovqe_n_cnots = []
        self.qovqe_n_gates = []
        self.qovqe_ansatz_circuit_per_distance = []