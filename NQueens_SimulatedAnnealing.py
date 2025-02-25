import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def GenerateBoard(N):
    """Generates NxN chessboard with N queens placed at random locations
    1: Queen located
    -1: No queen
    """
    board = -1 * np.ones([N, N], dtype=int)
    # Choose N distinct cells from the N*N cells.
    cells = random.sample(range(N * N), N)
    for cell in cells:
        row = cell // N
        col = cell % N
        board[row, col] = 1
    return board
def PrintReadableBoard(board):
    """Prints the chessboard in a human-friendly format"""
    board_dim = board.shape[0]

    for i in range(board_dim):
        line = ""
        for j in range(board_dim):
            if board[i][j] == 1:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

def ClassicalEnergy(N, board, A, B, C):
    """ Calculates the potential energy based on an Ising-like model with row, column & diagonal penalties and scaling factors A,B and C"""
    energy = 0.0

    # Row constraints
    for i in range(N):
        row_count = np.sum((1 + board[i, :]) // 2)
        energy += A * ((row_count - 1) ** 2)
    # Column constraints
    for j in range(N):
        col_count = np.sum((1 + board[:, j]) // 2)
        energy += B * ((col_count - 1) ** 2)
    # Main diagonals (NW-SE)
    for d in range(-N + 1, N):
        diag = np.diag(board, k=d)
        diag_count = np.sum((1 + diag) // 2)
        energy += C * (diag_count * (diag_count - 1))
    # Anti-diagonals (NE-SW)
    flipped = np.fliplr(board)
    for d in range(-N + 1, N):
        diag = np.diag(flipped, k=d)
        diag_count = np.sum((1 + diag) // 2)
        energy += C * (diag_count * (diag_count - 1))
    return energy

def SwapMove(board):
    """ Given a board with exactly N queens (cells with +1) and the rest -1,
        perform a swap move: select one cell that has a queen and one empty cell,
        then swap them. Returns the new board and the swapped positions. """
    board_new = board.copy()
    board_dim = board_new.shape[0]

    queen_positions = list(zip(*np.where(board == 1)))
    empty_positions = list(zip(*np.where(board == -1)))
    if not queen_positions or not empty_positions:
        return board_new, None

    pos_queen = random.choice(queen_positions)
    pos_empty = random.choice(empty_positions)

    board_new[pos_queen] = -1
    board_new[pos_empty] = 1

    return board_new, (pos_queen, pos_empty)

def RunSimulatedAnnealing(N, monte_carlo_steps, annealing_steps, initial_T, final_T, A, B, C):
    """Carry out simulated annealing to solve N-queens problem, applying swap move."""
    board = GenerateBoard(N)
    print(PrintReadableBoard(board))
    temperature_range = np.linspace(initial_T, final_T, annealing_steps)

    energies = []

    CurrentEnergy = ClassicalEnergy(N, board, A, B, C)

    for T in temperature_range:
        for _ in range(monte_carlo_steps):

            NewBoard = SwapMove(board)[0]
            NewEnergy = ClassicalEnergy(N, NewBoard, A, B, C)

            # Apply Metroplis
            random_number = np.random.uniform()
            delta_E = NewEnergy - CurrentEnergy
            if delta_E < 0 or random_number < np.exp(-delta_E / T):
                board = NewBoard
                CurrentEnergy = NewEnergy

            energies.append(CurrentEnergy)

    return board, energies, CurrentEnergy

FinalBoard, energies, FinalEnergy = RunSimulatedAnnealing(N=4, monte_carlo_steps=80, annealing_steps=50, initial_T=3, final_T=0.001, A=2, B=2, C=2)

print(PrintReadableBoard(FinalBoard))
print(FinalEnergy)

plt.figure(figsize=(8, 5))
plt.plot(energies, label="Simulated Annealing Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.title("Energy vs. Monte Carlo Steps")
plt.legend()
plt.show()
