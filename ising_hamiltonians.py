import numpy as np
import matplotlib.pyplot as plt

# Standard constants and matrices. J  is the graph defined by a 3-qubit problem given in Appendix 3
kappa = 0.5 # penalising factor (set in advance)
n = 3 # no of qubits
t = 5 # time at which to evaluate transverse field Ising model
tmax = 10 # total runtime of the algorithm

J = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])

h = np.array([[-1 + kappa],
              [-2 + kappa],
              [-1 + kappa]])

sigma_z = np.array([[1, 0],
                    [0, -1]])

sigma_x = np.array([[0, 1],
                    [1, 0]])

I_2 = np.eye(2)

#print(J[1][2])
#print(h[1])

def sigma_z_j(n,j):
    " Tensor product for sigma^z_j, where n is the total number of qubits in the system and the Pauli-Z gate is applied to the jth qubit "
    calc = np.eye(1) # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^z at the j-th position
    calc = np.kron(calc, sigma_z)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0 # Changes all instances of -0. to 0.
    return calc

def sigma_x_j(n,j):
    " Tensor product for sigma^x_j, where n is the total number of qubits in the system and the Pauli-X gate is applied to the jth qubit "
    calc = np.eye(1) # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^x at the j-th position
    calc = np.kron(calc, sigma_x)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0 # Changes all instances of -0. to 0.
    return calc

# Construct H_Ising, the Problem Hamiltonian, for the above matrix (graph) J and its "penalising matrix" h

def H_Ising(J, h):

    H_problem = np.zeros((2 ** n, 2 ** n))  # Size of final Hamiltonian

    for k in range(n):
        for j in range(k+1, n):
            if J[k, j] != 0:  # Only compute if J_kj is non-zero
                sigma_kz = sigma_z_j(n, k+1)
                sigma_jz = sigma_z_j(n, j+1)
                H_problem += J[k, j] * np.matmul(sigma_kz, sigma_jz)
                #print(J[k, j], sigma_kz, sigma_jz)

    for j in range(n):
        if h[j, 0] != 0:  # Only compute if h_j is non-zero
            sigma_jz = sigma_z_j(n, j+1)
            H_problem += h[j, 0] * sigma_jz


    return H_problem

#print("Problem Hamiltonian:")
#print(H_Ising(J,h))


def H_transverseField(J, h, t, tmax):
    A = 1 - t/tmax
    B = t/tmax
    H_evolve = np.zeros((2 ** n, 2 ** n))

    H_ising_term = H_Ising(J, h)
    H_interaction = B * H_ising_term

    for j in range(1, n+1):
        H_evolve -= A * sigma_x_j(n, j)

    H_t = H_evolve + H_interaction

    return H_t

print("Adiabatic Evolution Hamiltonian:")
print(H_transverseField(J, h, t, tmax))

