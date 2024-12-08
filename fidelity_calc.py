import numpy as np
import scipy
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# LaTeX Formatting

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm, amssymb, amsmath}'
# If necessary, specify the LaTeX installation path
# plt.rcParams['text.latex.unicode'] = True

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams['axes.labelweight'] = 'bold'       # X and Y axis labels
#plt.rcParams['axes.titleweight'] = 'bold'

bold_font = FontProperties(weight='bold')

# Standard constants and matrices.

kappa = 0.5  # penalising factor (set in advance)
n = 5  # no of qubits
tmax = [1, 5, 10, 30, 100]  # total runtime of the algorithm
q = 200  # number of timesteps
i = complex(0, 1)  # complex i

J = np.array([[0, 1, 1, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])

h = np.array([[-2 + kappa],
              [-3 + kappa],
              [-4 + kappa],
              [-1 + kappa],
              [-2 + kappa]])

sigma_z = np.array([[1, 0],
                    [0, -1]])

sigma_x = np.array([[0, 1],
                    [1, 0]])

I_2 = np.eye(2)


def sigma_z_j(n, j):
    " Tensor product for sigma^z_j, where n is the total number of qubits in the system and the Pauli-Z gate is applied to the jth qubit "
    calc = np.eye(1)  # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^z at the j-th position
    calc = np.kron(calc, sigma_z)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0  # Changes all instances of -0. to 0.
    return calc


def sigma_x_j(n, j):
    " Tensor product for sigma^x_j, where n is the total number of qubits in the system and the Pauli-X gate is applied to the jth qubit "
    calc = np.eye(1)  # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^x at the j-th position
    calc = np.kron(calc, sigma_x)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0  # Changes all instances of -0. to 0.
    return calc


def H_Ising(J, h):
    " Produces time-independent Ising matrix from graph J and penalising matrix h "

    H_problem = np.zeros((2 ** n, 2 ** n))  # Size of final Hamiltonian

    for k in range(n):
        for j in range(k + 1, n):
            if J[k, j] != 0:  # Only compute if J_kj is non-zero
                sigma_kz = sigma_z_j(n, k + 1)
                sigma_jz = sigma_z_j(n, j + 1)
                H_problem += J[k, j] * np.matmul(sigma_kz, sigma_jz)

    for j in range(n):
        if h[j, 0] != 0:  # Only compute if h_j is non-zero
            sigma_jz = sigma_z_j(n, j + 1)
            H_problem += h[j, 0] * sigma_jz

    return H_problem


#-------------------------------------------------------------------------------------------------------------------------------------------


# Initialise t=0 ground state
def initial_state(n):
    return np.ones(2 ** n) / np.sqrt(2 ** n)


psi_0 = initial_state(n)  # psi at t=0

# Encode maximum independent set - will write proper function for this

max_independent_set = np.zeros(2 ** n)
max_independent_set[19] = 1


# Create born rule function - this is for calculating success probability and fidelity
def born_rule(state, target_state):
    inner_product = np.vdot(target_state, state)  # Inner product <target_state|state>
    probability = np.abs(inner_product) ** 2  # Square of the absolute value
    return probability


# Calculate energy eigenvectors and eigenvalues after each evolution step

def energy(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value / q
    psi = psi_0

    timesteps = np.arange(0, tmax_value, step)

    ground_energies = np.zeros(len(timesteps))
    first_excited_energies = np.zeros(len(timesteps))
    energy_gap = np.zeros(len(timesteps))

    for i, t in enumerate(timesteps):
        # the following functions (A and B) are annealing schedules
        A = 1 - t / tmax_value
        B = t / tmax_value

        H_evolve = np.zeros((2 ** n, 2 ** n))

        H_ising_term = H_Ising(J, h)
        H_interaction = B * H_ising_term

        for j in range(1, n + 1):
            H_evolve -= A * sigma_x_j(n, j)

        H_t = H_evolve + H_interaction

        eigenvalues = np.sort(scipy.linalg.eigh(H_t, eigvals_only=True))

        ground_energies[i] = eigenvalues[0]
        first_excited_energies[i] = eigenvalues[1]

        # energy gap between ground state and first excited states
        energy_gap[i] = eigenvalues[1] - eigenvalues[0]

    return energy_gap, ground_energies, first_excited_energies

def calculate_fidelity(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value/q
    psi = psi_0
    fidelities = []

    timesteps = np.arange(0, tmax_value, step)

    for t in timesteps:
        # the following functions (A and B) are annealing schedules
        A = 1 - t/tmax_value
        B = t/tmax_value

        H_evolve = np.zeros((2 ** n, 2 ** n))

        H_ising_term = H_Ising(J, h)
        H_interaction = B * H_ising_term

        for j in range(1, n+1):
            H_evolve -= A * sigma_x_j(n, j)

        H_t = H_evolve + H_interaction

        e_vals, e_vectors = scipy.linalg.eigh(H_t)
        groundState = e_vectors[:,0]

        psi = np.matmul(scipy.linalg.expm(-1 * i * step * H_t), psi) # evolution of the state
        fidelity = born_rule(psi, groundState) # calculate fidelity

        fidelities.append(fidelity)

    return fidelities

# Fidelity as a function of t/tmax, for various tmax

plt.figure(figsize=(10,8))
colors = sns.color_palette("hls", len(tmax))

for tmax_value, color in zip(tmax, colors):
    success_probability = calculate_fidelity(J, h, psi_0, max_independent_set, tmax_value, q)

    x_values = np.arange(0, tmax_value, tmax_value/q) / tmax_value
    y_values = success_probability

    plt.plot(x_values, y_values, label=f'$t_{{max}}$ = {tmax_value}', color=color)

plt.xlabel('$t$/$t_{{max}}$')
plt.ylabel('Fidelity')
plt.legend(title = "Algorithm runtime")
#plt.savefig('fidelity_plot.svg', transparent=True)
plt.show()

# Instantaneous fidelity as a function of tmax

Palette = sns.color_palette("hls", 5)

plt.figure(figsize=(8, 6))
plt.ylim(0, 1.05)
plt.xlim(0, 53)

tmax_fidelities = np.arange(1, 50, 1)
fidelities_individual = np.zeros(len(tmax_fidelities))

for tmax in tmax_fidelities:
    fidelity_value = calculate_fidelity(J, h, ground_state=psi_0, target_state=max_independent_set, tmax_value=tmax, q=q)[-1]
    fidelities_individual[tmax-1] = fidelity_value

# plotting fidelity curve
plt.plot(tmax_fidelities, fidelities_individual, linewidth=3, color=Palette[3])

x_limits = plt.gca().get_xlim()
y_limits = plt.gca().get_ylim()

threshold_90 = 0.90
threshold_99 = 0.99

# critical indices for thresholds
critical_index_90 = np.argmin(np.abs(fidelities_individual - threshold_90))
critical_index_99 = np.argmin(np.abs(fidelities_individual - threshold_99))

min_fidelity = fidelities_individual.min()

# region where fidelity <= 0.90

plt.fill_between(
    tmax_fidelities[:critical_index_90+1],
    min_fidelity,  # Start filling from the x-axis
    fidelities_individual[:critical_index_90+1],
    color=Palette[0],
    alpha=0.2,
)

# region where fidelity <= 0.99

plt.fill_between(
    tmax_fidelities[critical_index_90:critical_index_99 + 1],  # X-axis range
    min_fidelity,  # Lower boundary (start from 0.90)
    fidelities_individual[critical_index_90:critical_index_99 + 1],  # Upper boundary (Fidelity curve)
    color=Palette[0],
    alpha=0.1,
)


plt.hlines(
    y=threshold_90,
    xmin=x_limits[0],  # Extend to the left edge of the graph
    xmax=tmax_fidelities[critical_index_90],  # Extend to the right edge of the graph
    color='#004D40',
    linestyle='--',
    linewidth=3
)
plt.hlines(
    y=threshold_99,
    xmin=x_limits[0],  # Extend to the left edge of the graph
    xmax=tmax_fidelities[critical_index_99],  # Extend to the right edge of the graph
    color=Palette[4],
    linestyle='--',
    linewidth=3,
)

plt.vlines(
    x=tmax_fidelities[critical_index_90],
    ymin=y_limits[0],  # Start at the bottom of the graph
    ymax=threshold_90,  # End at the threshold
    color="#004D40",
    linestyle='--',
    linewidth=3,
    label=r"$\mathbf{\mathcal{F} = 0.90}$" + "\n" + r"$\mathbf{t_{max} = " + str(tmax_fidelities[critical_index_90]) + "}$"
)

plt.vlines(
    x=tmax_fidelities[critical_index_99],
    ymin=y_limits[0],  # Start at the bottom of the graph
    ymax=threshold_99,  # End at the threshold
    color=Palette[4],
    linestyle='--',
    linewidth=3,
    label=r"$\mathbf{\mathcal{F} = 0.99}$" + "\n" + r"$\mathbf{t_{max} = " + str(tmax_fidelities[critical_index_99]) + "}$"
)


# Horizontal line at minimum possible fidelity
plt.axhline(
    y=min_fidelity,
    color='gray',
    linestyle='-.',
)

# Mark the point where Fidelity = Threshold
minimum_index = np.argmin(fidelities_individual)
t_90_index = np.argmax(fidelities_individual >= threshold_90)  # First index where Fidelity >= 0.90
t_90 = tmax_fidelities[t_90_index]  # Corresponding `tmax`


plt.scatter(tmax_fidelities[critical_index_99], fidelities_individual[critical_index_99], color='black', zorder=5)
plt.scatter(tmax_fidelities[critical_index_90], fidelities_individual[critical_index_90], color='black', zorder=5)
plt.scatter(tmax_fidelities[minimum_index], fidelities_individual[minimum_index], color='black', zorder=5)


plt.ylabel(r'$\mathbf{Fidelity, \mathcal{F}}$')
plt.xlabel(r'$\mathbf{t_{max}}$') # don't use \textbf{}
plt.legend(loc='center right', frameon=True, shadow=True)
plt.savefig("fidelity_inst.svg", bbox_inches='tight', transparent=True)
plt.tight_layout()
plt.show()

